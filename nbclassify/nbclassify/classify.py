# -*- coding: utf-8 -*-

"""Methods for image classification using artificial neural networks."""

import hashlib
import logging
import os
import sys

from cPickle import load
from pyfann import libfann

from .base import Common
from .data import Phenotyper
from .exceptions import *
from .functions import (combined_hash, get_childs_from_hierarchy,
    get_classification, get_codewords, get_config_hashables,
    get_phenotype_with_bowcode, get_bowcode_from_surf_features, 
    check_if_file_exists)

class ImageClassifier(Common):

    """Classify an image."""

    def __init__(self, config):
        super(ImageClassifier, self).__init__(config)
        self.error = 0.0001
        self.cache = {}
        self.roi = None

        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise ConfigurationError("classification hierarchy not set")

        # Get the taxon hierarchy.
        self.taxon_hr = self.get_taxon_hierarchy()

    def set_error(self, error):
        """Set the default maximum error for classification."""
        if not 0 < error < 1:
            raise ValueError("Error must be a value between 0 and 1" % error)
        self.error = error

    def set_roi(self, roi):
        """Set the region of interest for the image.

        If a region of interest is set, only that region is used for image
        processing. The ROI must be a ``(x, y, w, h)`` coordinates tuple.
        """
        if roi is not None:
            if len(roi) != 4:
                raise ValueError("ROI must be a list of four integers")
            for x in roi:
                if not (isinstance(x, int) and x >= 0):
                    raise ValueError("ROI must be a (x, y, w, h) tuple")
        self.roi = roi

    def get_classification_hierarchy_levels(self):
        """Return the list of level names from the classification hierarchy."""
        return [l.name for l in self.class_hr]

    def classify_image(self, im_path, ann_path, config, codebookfile=None):
        """Classify an image file and return the codeword.

        Preprocess and extract features from the image `im_path` as defined
        in the configuration object `config`, and use the features as input
        for the neural network `ann_path` to obtain a codeword.
        If necessary the 'codebookfile' is used to create the codeword.
        """
        for filename in [im_path, ann_path, codebookfile]:
            if filename != None:
                check_if_file_exists(filename)
        if 'preprocess' not in config:
            raise ConfigurationError("preprocess settings not set")
        if 'features' not in config:
            raise ConfigurationError("features settings not set")

        ann = libfann.neural_net()
        ann.create_from_file(str(ann_path))

        phenotype = self.get_phenotype(im_path, config)

        # If the SURF algorithm is applied, convert SURF features to BagOfWords-code.
        for name in sorted(vars(config.features).keys()):
            if name == 'surf':
                with open(codebookfile, "rb") as cb:
                    codebook = load(cb)
                phenotype = get_phenotype_with_bowcode(phenotype, codebook)

        logging.debug("Using ANN `%s`" % ann_path)
        codeword = ann.run(phenotype)
        return codeword

    def classify_with_hierarchy(self, image_path, ann_base_path=".", 
                                codebook_base_path=".", path=[], path_error=[], 
                                codebookfile=None):
        """Start recursive classification.

        Classify the image `image_path` with neural networks from the
        directory `ann_base_path`. The image is classified for each level
        in the classification hierarchy ``classification.hierarchy`` set in
        the configurations file. Each level can use a different neural
        network for classification; the file names for the neural networks
        are set in ``classification.hierarchy[n].ann_file``. Multiple
        classifications are returned if the classification of a level in
        the hierarchy returns multiple classifications, in which case the
        classification path is split into multiple classifications paths.

        Returns a pair of tuples ``(classifications, errors)``, the list
        of classifications, and the list of errors for each classification.
        Each classification is a list of the classes for each level in the
        hierarchy, top to bottom. The list of errors has the same dimension
        of the list of classifications, where each value corresponds to the
        mean square error of each classification.
        """
        levels = self.get_classification_hierarchy_levels()
        paths = []
        paths_errors = []

        if len(path) == len(levels):
            return ([path], [path_error])
        elif len(path) > len(levels):
            raise ValueError("Classification hierarchy depth exceeded")

        # Get the level specific configurations.
        level = conf = self.class_hr[len(path)]

        # Replace any placeholders in the ANN path.
        ann_file = level.ann_file
        for key, val in zip(levels, path):
            val = val if val is not None else '_'
            ann_file = ann_file.replace("__%s__" % key, val)

        class_errors, classes = self.get_levels_and_classes(path, level, 
                                 ann_base_path, ann_file, codebookfile, 
                                 codebook_base_path, image_path, conf)

        failed = self.create_info_messages(path, classes, level)
        if failed:
            return failed

        for class_, mse in zip(classes, class_errors):
            # Recurse into lower hierarchy levels.
            paths_, paths_errors_ = self.classify_with_hierarchy(image_path,
                ann_base_path, codebook_base_path, path+[class_], 
                path_error+[mse], codebookfile)

            # Keep a list of each classification path and their
            # corresponding errors.
            paths.extend(paths_)
            paths_errors.extend(paths_errors_)

        assert len(paths) == len(paths_errors), \
            "Number of paths must be equal to the number of path errors"
        return paths, paths_errors


    def get_levels_and_classes(self, path, level, ann_base_path, ann_file,
                               codebookfile, codebook_base_path, image_path,
                               conf):
        """Return the classes with their errors.
       
        Check if there are multiple classes for this level and
        return the classification for those classes with their mse.
        """
        # Get the class names for this node in the taxonomic hierarchy.
        level_classes = get_childs_from_hierarchy(self.taxon_hr, path)

        # Some levels must have classes set.
        if level_classes == [None] and level.name in ('genus','species'):
            raise ValueError("Classes for level `%s` are not set" % level.name)

        # No need to classify if there are no or only one class for current level.
        if level_classes == [None] or len(level_classes) == 1:
            classes = level_classes
            class_errors = [0.0]
        else:
            class_errors, classes = self.get_classes_and_errors(level_classes,
                 ann_base_path, ann_file, level, codebookfile, codebook_base_path,
                 image_path, conf)
        return class_errors, classes

    def get_classes_and_errors(self, level_classes, ann_base_path, ann_file, 
                               level, codebookfile, codebook_base_path, 
                               image_path, conf):
        """Return the classes with their errors to self.get_levels_and_classes."""
        # Get the codewords for the classes.
        class_codewords = get_codewords(level_classes)

        # Classify the image and obtain the codeword.
        ann_path = os.path.join(ann_base_path, ann_file)
        if not codebookfile and 'surf' in sorted(vars(conf.features).keys()):
            codebookfile = codebook_base_path + ann_file.replace('ann',
                                                    'tsv_codebook.file')
        codeword = self.classify_image(image_path, ann_path, conf, codebookfile)

        # Set the maximum classification error for this level.
        try:
            max_error = level.max_error
        except:
            max_error = self.error

        # Get the class name associated with this codeword.
        classes = get_classification(class_codewords, codeword, max_error)
        if classes:
            class_errors, classes = zip(*classes)
        else:
            class_errors = classes = []
        return class_errors, classes

    def create_info_messages(self, path, classes, level):
        """Print some info messages."""
        path_s = '/'.join([str(p) for p in path])

        # Return the classification if classification failed on current level.
        if len(classes) == 0:
            logging.debug("Failed to classify on level `%s` at node `/%s`" % (
                level.name,
                path_s)
            )
            return ([path], [path_error])
        elif len(classes) > 1:
            logging.debug("Branching in level `%s` at node '/%s' into `%s`" % (
                level.name,
                path_s,
                ', '.join(classes))
            )
        else:
            logging.debug("Level `%s` at node `/%s` classified as `%s`" % (
                level.name,
                path_s,
                classes[0])
            )

    def get_phenotype(self, im_path, config):
        """Return the phenotype of an image."""
        # Get the MD5 hash for the image.
        hasher = hashlib.md5()
        with open(im_path, 'rb') as fh:
            buf = fh.read()
            hasher.update(buf)

        # Get a hash that that is unique for this image/preprocess/features
        # combination.
        hashables = get_config_hashables(config)
        hash_ = combined_hash(hasher.hexdigest(),
            config.features, *hashables)

        if hash_ in self.cache:
            phenotype = self.cache[hash_]
        else:
            phenotyper = Phenotyper()
            phenotyper.set_image(im_path)
            if self.roi:
                phenotyper.set_roi(self.roi)
            phenotyper.set_config(config)
            phenotype = phenotyper.make()

            # Cache the phenotypes, in case they are needed again.
            self.cache[hash_] = phenotype
        return phenotype

