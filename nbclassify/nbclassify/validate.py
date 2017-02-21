# -*- coding: utf-8 -*-

"""Validation routines for artificial neural networks."""

from collections import Counter
import os

import numpy as np
from sklearn import cross_validation

from .base import Common
from .data import BatchMakeTrainData
from .training import BatchMakeAnn, TestAnn
import nbclassify.db as db

class Validator(Common):

    """Validate artificial neural networks.

    Must be used within a database session scope.
    """

    def __init__(self, config, cache_dir, temp_dir):
        """Constructor for the validator.

        Expects a configurations object `config`, the path to the directory
        where extracted features are cached `cache_dir`, and the path to the
        directory where temporary files are stored `temp_dir`.
        """
        super(Validator, self).__init__(config)
        self.set_cache_dir(cache_dir)
        self.set_temp_dir(temp_dir)
        self.aivolver_config_path = None

    def set_aivolver_config_path(self, path):
        """Set the path for the Aivolver configuration file `path`.

        Setting this value results in training with Aivolver.
        """
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.aivolver_config_path = path

    def set_cache_dir(self, path):
        """Set the path to the directory where features are cached."""
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.cache_dir = path

    def set_temp_dir(self, path):
        """Set the path to the directory where temporary files are stored.

        These files include training data, neural networks, and test results.
        """
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.temp_dir = path

    def k_fold_xval_stratified(self, k=3, autoskip=False):
        """Perform stratified K-folds cross validation.

        The number of folds `k` must be at least 2. The minimum number of
        members for any class cannot be less than `k`, or an AssertionError is
        raised. If `autoskip` is set to True, only the members for classes with
        at least `k` members are used for the cross validation.
        """
        session, metadata = db.get_session_or_error()

        # Will hold the score of each folds.
        scores = {}

        # Get a list of all the photo IDs in the database.
        samples = db.get_photos_with_taxa(session, metadata)
        photo_ids, classes, photo_count_min = self.__get_photo_ids(
            samples, k, autoskip)
        
        train_data, trainer, tester = self.__set_trainer_and_tester(
            photo_count_min)

        # Obtain cross validation folds.
        folds = cross_validation.StratifiedKFold(classes, k)
        result_dir = os.path.join(self.temp_dir, 'results')
        for i, (train_idx, test_idx) in enumerate(folds):
            self.__make_data_directories(result_dir, i)
            
            # Make train and test data for this fold.
            self.__make_data_subset(photo_ids, train_idx, train_data, 'train')
            self.__make_data_subset(photo_ids, test_idx, train_data, 'test')
            
            # Train neural networks on training data.
            trainer.batch_train(data_dir=self.train_dir, output_dir=self.ann_dir)
            
            scores = self.__get_scores(tester, scores)
            
        return scores
    
    def __get_photo_ids(samples, k, autoskip):
        # Get a list of the photo IDs and a list of the classes. The classes
        # are needed for the stratified cross validation.
        photo_ids = []
        classes = []
        for x in samples:
            photo_ids.append(x[0].id)
            tmp = np.array(x[1:]).astype(str)
            classes.append('_'.join(tmp))
            
        # Numpy features are needed for these.
        photo_ids = np.array(photo_ids)
        classes = np.array(classes)
        
        return self.__check_autoskip(photo_ids, classes, k, autoskip)
    
    def __check_autoskip(photo_ids, classes, k, autoskip):
        # Count the number of each class.
        class_counts = Counter(classes)

        if autoskip:
            # Create a mask for the classes that have enough members and remove
            # the photo IDs that don't have enough members.
            mask = []
            for i, c in enumerate(classes):
                if class_counts[c] >= k:
                    mask.append(i)

            photo_ids = photo_ids[mask]
            classes = classes[mask]
            photo_count_min = k
        else:
            for label, count in class_counts.items():
                assert count >= k, "Class {0} has only {1} members, which " \
                    "is too few. The minimum number of labels for any " \
                    "class cannot be less than k={2}. Use --autoskip to skip " \
                    "classes with too few members.".format(label, count, k)
            photo_count_min = 0
            
        return photo_ids, classes, photo_count_min
    
    def __set_trainer_and_tester(photo_count_min):
         # Train data exporter.
        train_data = BatchMakeTrainData(self.config, self.cache_dir)
        train_data.set_photo_count_min(photo_count_min)

        # Set the trainer.
        trainer = BatchMakeAnn(self.config)
        trainer.set_photo_count_min(photo_count_min)
        if self.aivolver_config_path:
            trainer.set_training_method('aivolver', self.aivolver_config_path)

        # Set the ANN tester.
        tester = TestAnn(self.config)
        tester.set_photo_count_min(photo_count_min)
        
        return train_data, trainer, tester
    
    def __make_data_directories(result_dir, i):
        # Make data directories.
        self.train_dir = os.path.join(self.temp_dir, 'train', str(i))
        self.test_dir = os.path.join(self.temp_dir, 'test', str(i))
        self.ann_dir = os.path.join(self.temp_dir, 'ann', str(i))
        self.test_result = os.path.join(result_dir, '{0}.tsv'.format(i))
        
        for path in (self.train_dir, self.test_dir, self.ann_dir, self.result_dir):
            if not os.path.isdir(path):
                os.makedirs(path)
    
    def __make_data_subset(photo_ids, index, train_data, mode):
        # Make subset of dataset.
        samples = photo_ids[index]
        train_data.set_subset(samples)
        
        # Save subset in correct directory.
        if mode == 'train':
            train_data.batch_export(self.train_dir)
        elif mode == 'test':
            train_data.batch_export(self.test_dir, self.train_dir)

    def __get_scores(tester, scores):
        # Calculate the score for this fold.
        tester.test_with_hierarchy(self.test_dir, self.ann_dir)
        tester.export_hierarchy_results(self.test_result)

        # List all level combinations.
        try:
            class_hr = self.config.classification.hierarchy
            hr = [level.name for level in class_hr]
        except:
            raise ConfigurationError("classification hierarchy not set")
        level_filters = []
        ranks = []
        for i in range(len(hr)):
            ranks.append(hr[i])
            level_filters.append(ranks)
        level_filters = tuple(level_filters)

        for filter_ in level_filters:
            correct, total = tester.get_correct_count(filter_)
            score = float(correct) / total
            
            filter_s = "/".join(filter_)
            if filter_s not in scores:
                scores[filter_s] = []
            scores[filter_s].append(score)
        
        return scores
    
