import functools
from typing import Union, Tuple, List, Optional, Dict
from collections import OrderedDict
import itertools
import copy
import json

import h5py
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from spanet.dataset.types import SpecialKey, NDArray, Batch, AssignmentTargets, Source, ArrayLike
from spanet.dataset.event_info import EventInfo
from spanet.dataset.inputs import create_source_input
from spanet.dataset.types import InputType
from spanet.dataset.regressions import regression_statistics

# The possible types for the limit index parameter.
TLimitIndex = Union[
    Tuple[float, float],
    List[float],
    float,
    np.ndarray,
    Tensor
]

# The format of a batch produced by this dataset
TBatch = Tuple[
    Tuple[Tuple[Tensor, Tensor], ...],
    Tensor,
    Tuple[Tuple[Tensor, Tensor], ...],
    Dict[str, Tensor],
    Dict[str, Tensor]
]

# todo: put this somewhere else... utils?
def make_json_safe(obj):
    """
    Recursively convert numpy types and arrays in a structure
    (dicts, lists, tuples, sets, etc.) into JSON-serializable types.
    """
    # handle dicts
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}

    # handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]

    # handle sets
    elif isinstance(obj, set):
        return [make_json_safe(x) for x in obj]  # convert to list for JSON

    # handle numpy scalars
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)

    # handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # fallback for anything else
    else:
        return obj


class JetReconstructionDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        event_info: Union[str, EventInfo],
        limit_index: TLimitIndex = 1.0,
        randomization_seed: int = 0,
        vector_limit: int = 0,
        partial_events: bool = True,
        pNN_reprocessing: dict = None,
        custom_mask: np.ndarray = None,
        clip_dict: dict = None,
        save_indices: bool = False,
        limit_index_sorting = True,
        shuffle_by_sample = False,
        global_balancing: dict = None
    ):
        """ A container class for reading in jet reconstruction datasets.

        Parameters
        ----------
        data_file : str
            HDF5 file containing the jet event data, see Notes section for structure information.
        event_info : str or EventInfo
            An EventInfo object which contains the symmetries for the event.
            Or the path of the yaml file where the event info is defined.
            See `feynman.dataset.EventInfo`
        limit_index : float in [-1, 1], tuple of floats, or array-like.
            If a positive float - limit the dataset to the first limit_index percent of the data.
            If a negative float - limit the dataset to the last |limit_index| percent of the data.
            If a tuple - limit the dataset to [limit_index[0], limit_index[1]] percent of the data.
            If array-like or tensor - limit the dataset to the specified indices.
        randomization_seed: int
            If set to a value greater than 0, randomize the order of the dataset. Applied before limit index.
        vector_limit: int
            Limit the event to a specific number of vectors.
        partial_events : bool
            Whether to allow training on partial events, not just complete events.
        pNN_reprocessing : dict
            A dictionary containing the pNN reprocessing parameters ('inpath': str, 'val': float, 'dtype': str)
        custom_mask : np.ndarray
            A numpy array filled with either:
             - boolean (simple mask, len = len n_events)
             - int (use case is for shuffling, len <= len n_events ) (not implemented yet)
        clip_dict : dict
            A dictionary containing information about which inputs to clip
        save_indices : bool
            Boolean indicating whether we ought to save the array from 'limit_index' or not...
        limit_index_sorting : bool
            Boolean indicating whether we ought to do 'np.sort(limit_index)' at end of function
            --> not advisable for large files
        shuffle_by_sample : bool [deprecated]
            Boolean indicating whether we ought to shuffle and split train/val by sample or not (default=False)
            Harcoded -> ought to really be used for debugging mainly...
        global_balancing : dict
            Config specifying how to balance our dataset --> a generalised version of 'shuffle_by_sample' which
            deals only with the sample array and addresses all unique samples

        """
        super(JetReconstructionDataset, self).__init__()

        self.data_file = data_file
        self.event_info: EventInfo = event_info
        self.clip_dict = clip_dict
        self.saved_indices = None
        self.balancing_info = None

        if isinstance(event_info, str):
            if ".ini" in event_info:
                self.event_info = EventInfo.read_from_ini(event_info)
            else:
                self.event_info = EventInfo.read_from_yaml(event_info)

        self.mean = None
        self.std = None

        print(f"Jet reconstruction init. dataset file:")
        print(f" - {self.data_file}")
        print(f" - rseed {randomization_seed}")
        print(f" - limit index: {limit_index}")

        with h5py.File(self.data_file, 'r') as file:
            # Get the first merged_momenta input to find the total number of events in the dataset.
            first_key = [
                name
                for name, input_type in self.event_info.input_types.items()
                if input_type in {InputType.Sequential, InputType.Relative}
            ][0]
            self.num_events = self.dataset(file, [SpecialKey.Inputs, first_key], SpecialKey.Mask).shape[0]
            # if custom_mask is not None:
            #     if ((custom_mask.dtype.kind == "b")
            #         and (custom_mask.shape[0] == self.num_events)):
            #         # is a mask
            #         self.num_events = np.sum(custom_mask)
            #     elif ((custom_mask.dtype.kind == "i")
            #          and (custom_mask.shape[0] <= self.num_events)):
            #          # is a shuffler (and/or a mask)
            #          self.num_events = custom_mask.shape[0]

            # Adjust limit index into a standard format.

            # if shuffle_by_sample:
            #     limit_index = self.compute_limit_index_by_sample(file, limit_index, randomization_seed, custom_mask=custom_mask, limit_index_sorting=limit_index_sorting)
            if global_balancing is not None:
                print(f"'global_balancing' initialisation of dataset")
                force_n = None
                if "force_n" in global_balancing and global_balancing["force_n"] is not None:
                    force_n = global_balancing["force_n"]

                limit_index, balancing_info = self.global_balancer(file, limit_index, global_balancing["selection"], randomization_seed, 
                    force_n=force_n, cut_to_min=global_balancing["balancing_cut"], limit_index_sorting=limit_index_sorting, 
                    custom_mask=custom_mask)
                
                self.balancing_info = balancing_info

                # in this case, we'd also like to save indices automatically
                if not save_indices:
                    print(f"Overriding 'save_indices' {save_indices} to 'True'")
                    save_indices = True

            elif custom_mask is None:
                limit_index = self.compute_limit_index(limit_index, randomization_seed, limit_index_sorting=limit_index_sorting)
            else:
                print(f"Applying custom mask as limit index")
                limit_index = self.compute_limit_index(np.where(custom_mask)[0], randomization_seed, 
                    limit_index_og=limit_index, limit_index_sorting=limit_index_sorting)
            
            if save_indices:
                self.saved_indices = limit_index
            print(f" - Saved indices: {self.saved_indices}")

            # Check if pNN reprocessing paramters are valid
            if pNN_reprocessing is not None:
                # Inpath being: 
                pNN_reprocessing_split = pNN_reprocessing['inpath'].split('/')
                pNN_reprocessing_group = pNN_reprocessing_split[0]
                pNN_reprocessing_key   = pNN_reprocessing_split[-1]
                reprocessing_ds = self.dataset(file, [SpecialKey.Inputs, pNN_reprocessing_group], pNN_reprocessing_key)
                print(f"Found valid pNN reprocessing dataset at {SpecialKey.Inputs}/{pNN_reprocessing_group}/{pNN_reprocessing_key}")
                print(f" - shape: {reprocessing_ds.shape}, dtype: {reprocessing_ds.dtype}")

            # Load source features from hdf5 file, processing them depending on their type.
            self.sources = OrderedDict((
                (input_name, create_source_input(self.event_info, file, input_name, self.num_events, limit_index, pNN_reprocessing, clip_dict))
                for input_name in self.event_info.input_names
            ))

            # Compute the jet offsets for different input sources if we are reconstructing more than one type of object.
            self.source_offsets = torch.tensor([
                dataset.max_vectors()
                for name, dataset in self.sources.items()
                if dataset.reconstructable
            ])
            self.source_offsets = torch.nn.functional.pad(self.source_offsets, (1, 0), value=0)
            self.source_offsets = torch.cumsum(self.source_offsets, 0)[:-1]

            # Load various types of targets.
            self.assignments = self.load_assignments(file, limit_index)
            self.regressions, self.regression_types = self.load_regressions(file, limit_index)
            self.classifications = self.load_classifications(file, limit_index)
            self.custom_weights = self.load_custom_weights(file, limit_index)

            # Update size information after loading and limiting dataset.
            self.num_events = limit_index.shape[0]
            self.num_vectors = sum(source.num_vectors() for source in self.sources.values())

            print(f"Index Range: {limit_index[0]}...{limit_index[-1]}")

        # Optionally remove any events where any of the targets are missing.
        if not partial_events:
            self.limit_dataset_to_full_events()
            print(f"Training on Full Events only.")

        # Optionally limit the dataset to a specific number of jets.
        if vector_limit > 0:
            self.limit_dataset_to_jet_count(vector_limit)
        
        # Apply custom mask if given (this is already done via compute_limit_index)
        # if custom_mask is not None:
        #     print(f"Apply custom event mask")
            # copy otherwise it remains in 'mmap' mode...
            # self.limit_dataset_to_mask(custom_mask.copy())

    @staticmethod
    def dataset(hdf5_file: h5py.File, group: List[str], key: str) -> h5py.Dataset:
        group_string = "/".join(group)
        key_string = "/".join(group + [key])
        if key in hdf5_file[group_string]:
            return hdf5_file[key_string]
        else:
            raise KeyError(f"{key} not found in group {group_string}")

    def global_balancer(
        self, file: h5py.File, limit_index: TLimitIndex, conf: dict, 
        randomization_seed: int, force_n: int = None, cut_to_min: bool = True,
        limit_index_sorting: bool = True, custom_mask: np.ndarray = None,
        verbose: bool = True
    ):
        '''
            Given some file with some properties per event, e.g. sample, year, that we'd like
            to equalise, do so automatically...
            - file (h5py.File): opened file object
            - limit_index (tuple or float): what to give to each subcategory --> since we'd like
                exactly the same numbers also for validation... (otherwise gets complicated..)
            - conf (dict), information about the independent sortings e.g.,
                {'sample':{'list':[[47],[51],[55],[59],[63]], 'inpath':'WEIGHTS/EVENT/sample', 'method':None}}
                - 'list' (list or dict) of groupings, which uses some reference array to s
                where 'inpath' is compulsory if 'method' is None, else we have hardcoded 
        '''
        if verbose: print(f"GLOBAL_BALANCER VERBOSE OUTPUT:")

        if not cut_to_min:
            print(f"Warning! 'cut_to_min'={cut_to_min}, meaning we're just shuffling given some selection")
            print(f" - (no balancing going on...)")

        random_state = None
        if randomization_seed > 0:
            if verbose: print(f" - getting random state using seed {randomization_seed}")
            random_state = np.random.RandomState(seed=randomization_seed)
        
        # 1. Get all masks for each category
        if verbose: print(f" - getting masks for each category")
        groupings = {} # e.g., {'year':{'2016':..}, 'sample':{'zp500w4':..,'tt':..},..}
        for name, grouping in conf.items():
            if verbose:
                print(f" - - {name}:")
                for tk,tv in grouping.items(): print(f" - - - {tk}: {tv}")

            if grouping["method"] is None:
                # comparator = file[grouping["inpath"]]
                comparator = self.dataset(file, grouping["inpath"].split("/")[:-1], grouping["inpath"].split("/")[-1])
                # use_array = self.dataset(file, ["WEIGHTS/EVENT"], "sample")
                if verbose: print(f" - - - got comparator: {comparator.shape}, {comparator[0:5]}, {comparator}")
            # elif grouping["method"] == "num_tops":
                # would be hardcoded... but just an example
                # would make comparator an array of number of tops per event
                # though , be careful if we're clipping jets -> affects the n. tops... 
                # which i don't have been loaded in yet...
            else:
                raise NotImplementedError(f"Not implemented grouping method {grouping['method']}")

            # get masks using comparator
            masks = {}
            if "list" in grouping:
                for i, subgroup in enumerate(grouping["list"]):
                    if isinstance(grouping["list"], dict):
                        subgroup_name = subgroup
                        subgroup_value = grouping["list"][subgroup]
                    elif isinstance(grouping["list"], list):
                        subgroup_value = subgroup
                        # subgroup_name = f"{name}_" + "_".join(subgroup)
                        subgroup_name = "_".join(subgroup) # got rid of {name} since it'd be double..
                    else:
                        raise ValueError(f"Can't handle grouping 'list' type of {grouping['list']}")
                    
                    # use_mask = np.isin(subgroup_value, comparator)
                    use_mask = np.isin(comparator, subgroup_value)
                    if verbose: 
                        print(f" - - - - getting mask for {subgroup_name} using {subgroup_value}")
                        print(f" - - - - - comparator: {comparator[0:5]}, mask: {use_mask[0:5]}")

                    if custom_mask is not None:
                        # in case we made like variable cleaning cuts prior for example
                        use_mask = use_mask & custom_mask
                    masks[subgroup_name] = use_mask
            else:
                # assumes unique values are the subgroups we'd like...
                subgroups = np.unique(comparator)
                for unique_val in subgroups:
                    # subgroup_name = f"{name}_{unique_val}"
                    subgroup_name = f"{unique_val}" # got rid of {name} since it'd be double..
                    subgroup_value = [unique_val]
                    # use_mask = np.isin(subgroup_value, comparator)
                    use_mask = np.isin(comparator, subgroup_value)
                    if custom_mask is not None:
                        # in case we made like variable cleaning cuts prior for example
                        use_mask = use_mask & custom_mask
                    masks[subgroup_name] = use_mask

            # groupings.append(copy.deepcopy(masks))
            groupings[name] = copy.deepcopy(masks)

        # 2. Now get all the combinations
        print(f" - getting all combinations")
        categories = list(groupings.keys())
        label_lists = [list(groupings[c].keys()) for c in categories]
        all_cats = {}
        min_events = None
        for combo in itertools.product(*label_lists):

            name = "_".join(f"{cat}_{label}" for cat, label in zip(categories, combo))
            combo_mask = np.logical_and.reduce([
                groupings[cat][label] for cat, label in zip(categories, combo)
            ])
            pass_combo_mask = np.sum(combo_mask)
            if min_events is None:
                min_events = pass_combo_mask
            elif pass_combo_mask < min_events:
                min_events = pass_combo_mask
            print(f" - - combo {combo}: {pass_combo_mask}")

            all_cats[name] = {
                'mask':combo_mask,
                'n_events_pre':pass_combo_mask
            }

        # Moving this --> it doesn't make sense if we do compute_limit_index after
        # if force_n is not None and force_n > min_events:
        #     print(f"Warning! 'force_n' is {force_n}, which is > min events {min_events}")
        #     print(f" - (should be smaller)")
        #     min_events = min_events
        # elif force_n is not None and force_n < min_events:
        #     min_events = force_n
        
        # 3. Permute & cut to first n...
        total_limit_index = []
        if verbose: print(f" - cutting on categories")
        for cat in all_cats:
            
            cat_idx = np.where(all_cats[cat]["mask"])[0]
            temp_mask = np.full(cat_idx.shape[0], True, dtype=bool)
            if verbose: 
                print(f" - - {cat}, {cat_idx[0:5], cat_idx.shape[0]}")
            if cut_to_min:
                temp_mask[min_events:] = False
            if random_state is not None:
                # shuffling beforehand so that we're not just
                # cutting out the first N events in case that's not properly
                # shuffled somehow...
                temp_mask = random_state.permutation(temp_mask)

            cat_idx = cat_idx[temp_mask]
            if verbose: print(f" - - - post-mask: {cat_idx[0:5], cat_idx.shape[0]}")

            all_cats[cat]['n_events_post_min_cut'] = cat_idx.shape[0]
            # this part is primarily if we want to use dataset_limit & train_validation_split..
            # tbh, not sure how to then extend this further for the test sample...
            # i.e., does limit_index = limit_index[lower_index:upper_index] per cat
            limit_index_by_cat = self.compute_limit_index(
                cat_idx, randomization_seed, limit_index_og=limit_index
            )
            all_cats[cat]['n_events_post_limit_index'] = limit_index_by_cat.shape[0]

            # todo: should we be worried about shuffling here?
            if force_n is not None and force_n > limit_index_by_cat.shape[0]:
                print(f"Warning! 'force_n' is {force_n}, which is > n events in cat {cat} ({limit_index_by_cat.shape[0]})")
                print(f" - (should be smaller)")
            elif force_n is not None and force_n < limit_index_by_cat.shape[0]:
                limit_index_by_cat = limit_index_by_cat[:force_n]
            all_cats[cat]['n_events_post_force_n'] = limit_index_by_cat.shape[0]

            total_limit_index.append(limit_index_by_cat)
        
        # 4. Concatenate all the categories --> shuffle/sort if specified
        total_limit_index = np.concatenate(total_limit_index)

        # save info
        events_by_category = {}
        for k in all_cats:
            # why is this so harcoded?
            save_info = {}
            for info in all_cats[k]:
                if info == "mask": continue
                save_info[info] = all_cats[k][info]
            events_by_category[k] = copy.deepcopy(save_info)

            # events_by_category[k] = {
            #     "n_events_pre":all_cats[k]["n_events_pre"],
            #     "n_events_post_min_cut":all_cats[k]["n_events_post_min_cut"],
            #     "n_events_post_limit_index":all_cats[k]["n_events_post_limit_index"]
            # }
            
        # assumes we won't make jet limit cuts / n top limit cuts or anything..
        events_by_category["total"] = total_limit_index.shape[0]
        events_by_category["total_randomized"] = random_state is None
        events_by_category["total_sorted"] = limit_index_sorting

        if random_state is not None:
            ret = random_state.permutation(total_limit_index)
        else:
            print(f"Warning, dataset is stacked from cat to cat!!! Should ideally shuffle!!!")
            ret = total_limit_index

        if limit_index_sorting:
            # should ideally not sort if we have a dataset that's stacked
            print(f"Warning, sorting idx of dataset that may have been stacked!!! Should ideally not sort!!!")
            ret = np.sort(ret)

        return ret, events_by_category
            

    def compute_limit_index_by_sample(self, file: h5py.File, limit_index: TLimitIndex, randomization_seed: int, 
        custom_mask: np.ndarray = None,
        limit_index_sorting: bool = True):
        '''
            Function to wrap around 'compute_limit_index' by selecting the indices by sample
            -- Such that, if we have samples A, B, C; each of nevents (1000,1000,1000), that the combined
            dataset's splitting does not create unequal numbers of A,B,C
            -- I.e., shuffling&masking per sample
            -- Will only create equal numbers if A,B,C (post-custom-mask) have equal numbers
            -- Puts no real requirement on the numbers of events of each sample

            Note: very specific use-case; mainly for debugging...
        '''
        print(f"Constructing limit index by sample")

        # 1. get sample array -> 2. loop and shuffle by sample -> 3. collate total indices
        use_array = self.dataset(file, ["WEIGHTS/EVENT"], "sample") # hardcoded...
        by_sample = np.unique(use_array)

        # collect 
        total_limit_index = []
        for sample_id in by_sample:
            print(f" - sample {sample_id}")
            print(f" - - limit index: {limit_index}")

            use_mask = use_array == sample_id
            print(f" - - n possible events (sample): {np.sum(use_mask)}")
            if custom_mask is not None:
                use_mask = use_mask & custom_mask
            if not (np.sum(use_mask) > 0):
                print(f"Warning: shuffling by sample but sample:{sample_id} has no events")
                continue
            print(f" - - n possible events (sample & mask): {np.sum(use_mask)}")
            limit_index_by_sample = self.compute_limit_index(np.where(use_mask)[0], randomization_seed, 
                    limit_index_og=limit_index)
            print(f" - - - limit indices: {limit_index_by_sample.shape[0]}")
            total_limit_index.append(limit_index_by_sample)

        total_limit_index = np.concatenate(total_limit_index)

        if randomization_seed > 0:
            random_state = np.random.RandomState(seed=randomization_seed)
            ret = random_state.permutation(total_limit_index)
        else:
            print(f"Warning, limit indices by sample")
            print(f" --> meaning each batch might not be representative (if also no sort)")
            # limit_index = limit_index[lower_index:upper_index]
            # don't need to do this since this is done per sample
            ret = total_limit_index
        if limit_index_sorting:
            return np.sort(ret)
        else:
            return ret
      


    def compute_limit_index(self, limit_index: TLimitIndex, randomization_seed: int,
        limit_index_og: Optional[TLimitIndex] = None,
        limit_index_sorting: bool = True) -> NDArray[np.int64]:
        """ Take subsection of the data for training / validation

        Parameters
        ----------
        limit_index : float in [-1, 1], tuple of floats, or array-like
            If a positive float - limit the dataset to the FIRST limit_index percent of the data
            If a negative float - limit the dataset to the LAST |limit_index| percent of the data
            If a tuple - limit the dataset to [limit_index[0], limit_index[1]] percent of the data
            If array-like or tensor - limit the dataset to the specified indices.
        randomization_seed: int
            If randomization_seed is non-zero, then we will first shuffle the indices in a deterministic manner
            before taking the subset defined by `limit_index`.
        limit_index_og : float in [-1, 1], tuple of floats
            Same as limit_index --> but ONLY TO BE USED when custom_mask has been applied
            such that we apply the same sort of 'limiting' of the dataset to some percent

        Returns
        -------
        np.ndarray or torch.Tensor
        """
        print(f"COMPUTING LIMIT INDEX: {limit_index}")
        # In the float case, we just generate the list with the appropriate bounds
        if isinstance(limit_index, float):
            limit_index = (0.0, limit_index) if limit_index > 0 else (1.0 + limit_index, 1.0)
            print(f" - converting to float --> {limit_index} ")

        # In the list / tuple case, we want a contiguous range
        if isinstance(limit_index, (list, tuple)):
            lower_index = int(round(limit_index[0] * self.num_events))
            upper_index = int(round(limit_index[1] * self.num_events))

            if randomization_seed > 0:
                random_state = np.random.RandomState(seed=randomization_seed)
                limit_index = random_state.permutation(self.num_events)
            else:
                limit_index = np.arange(self.num_events)

            print(f" - converting to array {limit_index} between {(lower_index, upper_index)}")
            limit_index = limit_index[lower_index:upper_index]

        # Convert to numpy array for simplicity
        elif isinstance(limit_index, (Tensor, np.ndarray)):
            if isinstance(limit_index, Tensor):
                limit_index = limit_index.numpy()
            # wow turns out that shuffling wasn't even happening in the end
            # if i'd given a mask...
            if randomization_seed > 0:
                print(f"APPLYING RANDOMIZATION TO A MASK:")
                # should the random state only be applied for training
                # or is val fine too? should be fine right?
                random_state = np.random.RandomState(seed=randomization_seed)
                limit_index = random_state.permutation(limit_index)
                print(f" - RANDOMISED: {limit_index}")
                # return limit_index
            if limit_index_og is not None:
                print(f"APPLYING LIMIT_INDEXING ({limit_index_og}) TO MASKED DATASET")
                if isinstance(limit_index_og, float):
                    print(f" - float case")
                    limit_index_og = (0.0, limit_index_og) if limit_index_og > 0 else (1.0 + limit_index_og, 1.0)
                # if isinstance(limit_index, (list, tuple)): 
                # ... (was i using the WRONG FUCKING ONE? yes, but, it didn't matter -> because i hadn't applied a mask...
                if isinstance(limit_index_og, (list, tuple)):
                    print(f" - tuple case")
                    lower_index = int(round(limit_index_og[0] * limit_index.shape[0]))
                    upper_index = int(round(limit_index_og[1] * limit_index.shape[0]))
                    limit_index = limit_index[lower_index:upper_index]
                # print(limit_index.shape[0])
        
        if limit_index_sorting:
            # Make sure the resulting index array is sorted for faster loading.
            return np.sort(limit_index)
        else:
            return limit_index

    def load_assignments(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Dict[str, Tuple[Tensor, Tensor, Tensor]]:
        """ Load target indices for every defined target

        Parameters
        ----------
        hdf5_file: h5py.File
            HDF5 file containing the event.
        limit_index: array or Tensor
            The limiting array for selecting a subset of dataset for this object.

        Returns
        -------
        OrderedDict: str -> (Tensor, Tensor)
            A dictionary mapping the target name to the target indices, mask and weight.
        """
        targets = OrderedDict()
        for event_particle, daughter_particles in self.event_info.product_particles.items():
            target_data = torch.empty(len(daughter_particles), self.num_events, dtype=torch.int64)

            for index, daughter in enumerate(daughter_particles):
                dataset = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], daughter)
                dataset.read_direct(target_data[index].numpy())

                if self.clip_dict is not None:
                    # get source name
                    source_idx = daughter_particles.sources[index]
                    source_name = list(self.sources.keys())[source_idx]
                    # if source name 
                    if f"{source_name}:MASK" in self.clip_dict:
                        # basically if we're clipping number of object
                        # can't have any assignments ≥ the assignment
                        # >= because indexing starts at 0 !
                        print(f"LOADING {event_particle} -> {daughter}")
                        print(f" - MASKING ASSIGNMENTS >= {self.clip_dict[f'{source_name}:MASK']['upper_bound']}")
                        affected_mask = target_data[index] >= self.clip_dict[f"{source_name}:MASK"]["upper_bound"]
                        affected_indices = np.flatnonzero(affected_mask)
                        target_data[index][affected_indices] = -1

            # Offset the targets if they are not global targets
            for index, source in enumerate(daughter_particles.sources):
                if source >= 0:
                    target_data[index] += self.source_offsets[source] * (target_data[index] >= 0)

            target_data = target_data.transpose(0, 1)

            # Either load an explicit mask or generate a mask based on the targets
            try:
                target_mask = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], SpecialKey.Mask)
                target_mask = torch.from_numpy(target_mask[:]).bool()
            except KeyError:
                target_mask = (target_data >= 0).all(1)

            # Either load an explicit weight or generate ones weights
            try:
                target_weight = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], SpecialKey.Weight)
            except KeyError:
                print("Warning: no target weights in the dataset, creating ones weights")
                target_weight = torch.ones_like(target_mask, dtype=float)

            target_data = target_data[limit_index]
            target_mask = target_mask[limit_index]
            target_weight = target_weight[limit_index]

            targets[event_particle] = (target_data, target_mask, target_weight)

        return targets

    def tree_key_data(self, hdf5_file: h5py.File, limit_index, root, group, index):
        key = "/".join((*group, index))
        data = self.dataset(hdf5_file, [root, *group], index)
        data = torch.from_numpy(data[:][limit_index])
        return key, data

    def load_regressions(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Tuple[Dict[str, Tensor], Dict[str, str]]:
        tree_key_data = functools.partial(self.tree_key_data, hdf5_file, limit_index, SpecialKey.Regressions)
        targets = OrderedDict()
        types = OrderedDict()

        for target in self.event_info.regressions[SpecialKey.Event]:
            key, data = tree_key_data([SpecialKey.Event], target.name)
            targets[key] = data
            types[key] = target.type

        for particle in self.event_info.event_particles:
            for target in self.event_info.regressions[particle][SpecialKey.Particle]:
                key, data = tree_key_data([particle, SpecialKey.Particle], target.name)
                targets[key] = data
                types[key] = target.type

            for daughter in self.event_info.product_particles[particle]:
                for target in self.event_info.regressions[particle][daughter]:
                    key, data = tree_key_data([particle, daughter], target.name)
                    targets[key] = data
                    types[key] = target.type

        return targets, types

    def load_classifications(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Dict[str, Tensor]:
        tree_key_data = functools.partial(self.tree_key_data, hdf5_file, limit_index, SpecialKey.Classifications)

        targets = OrderedDict()

        def add_target(key, value):
            targets[key] = value

        for target in self.event_info.classifications[SpecialKey.Event]:
            add_target(*tree_key_data([SpecialKey.Event], target))

        for particle in self.event_info.product_particles:
            for target in self.event_info.classifications[particle][SpecialKey.Particle]:
                add_target(*tree_key_data([particle, SpecialKey.Particle], target))

            for daughter in self.event_info.product_particles[particle]:
                for target in self.event_info.classifications[particle][daughter]:
                    add_target(*tree_key_data([particle, daughter], target))

        return targets
    
    def load_custom_weights(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Tensor:
        '''
            Function to load in event weights from:
            https://github.com/guanfacin24/SPANet/tree/dev
        '''
        # Avoids having to load in if not specified...
        if self.event_info.custom_weights is None:
            return None
        weights = torch.from_numpy(
            np.ones_like(limit_index, dtype = float)
        )
        weight_types = self.event_info.custom_weights[SpecialKey.Event]
        num_weights = len(weight_types)
        if num_weights > 1:
            print(
                "More than one custom event weight type specified\n"
                "Weights will be multiplied"
            )

        for weight in weight_types:
            try:
                weights *= torch.from_numpy(
                    hdf5_file[SpecialKey.CustomWeights][SpecialKey.Event][weight][limit_index]
                )
            except KeyError: continue

        return weights

    def compute_source_statistics(
            self,
            mean: Optional[Dict[str, Tensor]] = None,
            std: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """ Compute the mean and standard deviation of features with normalization enabled in the event file.

        Parameters
        ----------
        mean: Tensor, optional
        std: Tensor, optional
            Give existing values for mean and standard deviation to set this value
            dataset's statistics to those values. This is especially useful for
            normalizing the validation and testing datasets with training statistics.

        Returns
        -------
        (Tensor, Tensor)
            The new mean and standard deviation for this dataset.
        """
        if mean is None:
            mean = OrderedDict()
            std = OrderedDict()

            for input_name, source in self.sources.items():
                mean[input_name], std[input_name] = source.compute_statistics()

        self.mean = mean
        self.std = std

        return mean, std

    def compute_regression_statistics(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """ Compute the target regression statistics

        Returns
        -------
        (Dict[str, Tensor], Dict[str, Tensor])
            The mean and standard deviation for existing regression values.
        """
        regression_means = OrderedDict()
        regression_stds = OrderedDict()

        for key, value in self.regressions.items():
            if value is None:
                continue

            mean, std = regression_statistics(self.regression_types[key])(value)
            regression_means[key] = mean
            regression_stds[key] = std

        return regression_means, regression_stds

    def compute_classification_class_counts(self) -> Dict[str, int]:
        return OrderedDict((
            (key, value.max().item() + 1)
            for key, value in self.classifications.items()
            if value is not None
        ))

    def compute_particle_balance(self):
        # Extract just the mask information from the dataset.
        masks = torch.stack([target[1] for target in self.assignments.values()])

        eq_class_counts = {}
        num_targets = masks.shape[0]
        full_targets = frozenset(range(num_targets))

        # Find the count for every equivalence class in the masks.
        for eq_class in self.event_info.event_equivalence_classes:
            eq_class_count = 0

            for positive_target in eq_class:
                negative_target = full_targets - positive_target

                # Note that we must ensure that every sample is assigned exactly one equivalence class.
                # So we have to make sure that the ONLY target present in the one we want.
                positive_target = masks[list(positive_target), :].all(0)
                negative_target = masks[list(negative_target), :].any(0)

                targets = positive_target & ~negative_target
                eq_class_count += targets.sum().item()

            eq_class_counts[eq_class] = eq_class_count + 1

        # Compute the effective class count
        # https://arxiv.org/pdf/1901.05555.pdf
        beta = 1 - (10 ** -np.log10(masks.shape[1]))
        eq_class_weights = {key: (1 - beta) / (1 - (beta ** value)) for key, value in eq_class_counts.items()}
        target_weights = {target: weight for eq_class, weight in eq_class_weights.items() for target in eq_class}

        # Convert these target weights into a bit-mask indexed tensor
        norm = sum(eq_class_weights.values())
        index_tensor = 2 ** np.arange(num_targets)
        target_weights_tensor = torch.zeros(2 ** num_targets)

        for target, weight in target_weights.items():
            index = index_tensor[list(target)].sum()
            target_weights_tensor[index] = len(eq_class_weights) * weight / norm

        return torch.from_numpy(index_tensor), target_weights_tensor

    def compute_vector_balance(self):
        max_vectors = self.num_vectors.max()
        min_vectors = self.num_vectors.min()

        class_count = torch.bincount(self.num_vectors, minlength=max_vectors + 1)

        # Compute the effective class count
        # https://arxiv.org/pdf/1901.05555.pdf
        beta = 1 - (1 / self.num_vectors.shape[0])
        vector_class_weights = (1 - beta) / (1 - (beta ** class_count))
        vector_class_weights[torch.isinf(vector_class_weights)] = 0
        vector_class_weights = (max_vectors - min_vectors + 1) * vector_class_weights / vector_class_weights.sum()

        return vector_class_weights

    def compute_classification_balance(self):
        def compute_effective_counts(targets):
            beta = 1 - (1 / targets.shape[0])
            vector_class_weights = (1 - beta) / (1 - (beta ** torch.bincount(targets)))
            vector_class_weights[torch.isinf(vector_class_weights)] = 0
            vector_class_weights = vector_class_weights.shape[0] * vector_class_weights / vector_class_weights.sum()

            return vector_class_weights

        return OrderedDict((
            (key, compute_effective_counts(value))
            for key, value in self.classifications.items()
            if value is not None
        ))

    def limit_dataset_to_mask(self, event_mask: Tensor):
        for input_name, source in self.sources.items():
            source.limit(event_mask)

        for key in self.assignments:
            assignments, masks, weights = self.assignments[key]

            assignments = assignments[event_mask].contiguous()
            masks = masks[event_mask].contiguous()
            weights = weights[event_mask].contiguous()

            self.assignments[key] = (assignments, masks, weights)

        for key, regressions in self.regressions.items():
            self.regressions[key] = regressions[event_mask]

        for key, classifications in self.classifications.items():
            self.classifications[key] = classifications[event_mask]

        self.num_events = event_mask.sum().item()
        self.num_vectors = sum(source.num_vectors() for source in self.sources.values())

        if self.saved_indices is not None:
            self.saved_indices = self.saved_indices[event_mask]

    def limit_dataset_to_partial_events(self):
        vector_masks = torch.stack([target[1] for target in self.assignments.values()])
        non_empty_events = vector_masks.any(0)
        self.limit_dataset_to_mask(non_empty_events)

    def limit_dataset_to_full_events(self):
        vector_masks = torch.stack([target[1] for target in self.assignments.values()])
        full_events = vector_masks.all(0)
        self.limit_dataset_to_mask(full_events)

    def limit_dataset_to_jet_count(self, jet_count):
        self.limit_dataset_to_mask(self.num_vectors == jet_count)
    
    def save_indices_to_file(self, outfile):
        '''
            if dataset has indices saved (if we applied our own train/val splitting)
            --> save it
        '''
        if self.saved_indices is None:
            print(f"No indices recorded, nothing to save")
            return False
        np.save(outfile, self.saved_indices)
        return True

    
    def save_balancing_info_to_file(self, outfile):
        ''' 
            if dataset has been balanced via global_balancer --> save this info...
        '''
        if self.balancing_info is None:
            print(f"No 'balancing_info' found --> nothing to save")
            return False
        
        with open(outfile, "w") as file:
            json.dump(make_json_safe(self.balancing_info), file, indent=4)
        return True


    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, item) -> Batch:
        sources = tuple(
            source[item]
            for source in self.sources.values()
        )

        assignments = tuple(
            AssignmentTargets(assignment[item], mask[item], weight[item])
            for assignment, mask, weight in self.assignments.values()
        )

        regressions = {
            key: value[item]
            for key, value in self.regressions.items()
            if value is not None
        }

        classifications = {
            key: value[item]
            for key, value in self.classifications.items()
            if value is not None
        }

        return Batch(
            sources,
            self.num_vectors[item],
            assignments,
            regressions,
            classifications,
            item
        )
