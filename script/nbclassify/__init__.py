#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import sys

import cv2
import features as ft
import numpy as np
from pyfann import libfann
import sqlalchemy.orm as orm
from sqlalchemy.ext.automap import automap_base

from exceptions import *

class Struct(argparse.Namespace):
    """Return a dictionary as an object."""

    def __init__(self, d):
        for key, val in d.iteritems():
            if isinstance(val, (list, tuple)):
                setattr(self, str(key), [self.__class__(x) if \
                    isinstance(x, dict) else x for x in val])
            else:
                setattr(self, str(key), self.__class__(val) if \
                    isinstance(val, dict) else val)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

class Common(object):
    def __init__(self, config):
        self.set_config(config)

    def set_config(self, config):
        """Set the YAML configurations object."""
        if not isinstance(config, Struct):
            raise TypeError("Configurations object must be of type Struct, not %s" % type(config))

        try:
            path = config.preprocess.segmentation.output_folder
        except:
            path = None
        if path and not os.path.isdir(path):
            logging.error("Found a configuration error")
            raise IOError("Cannot open %s (no such directory)" % path)

        self.config = config

    def get_codewords(self, classes, on=1, off=-1):
        """Return codewords for a list of classes."""
        n =  len(classes)
        codewords = {}
        for i, class_ in enumerate(sorted(classes)):
            cw = [off] * n
            cw[i] = on
            codewords[class_] = cw
        return codewords

    def get_classification(self, codewords, codeword, error=0.01, on=1.0):
        """Return the human-readable classification for a codeword.

        Each bit in the codeword `codeword` is compared to the `on` bit in
        each of the codewords in `codewords`, which is a dictionary of the
        format ``{class: codeword, ..}``. If the mean square error for a bit
        is less than or equal to `error`, then the corresponding class is
        assigned to the codeword. So it is possible that a codeword is
        assigned to multiple classes.

        The result is returned as a sorted list of 2-tuples ``[(error,
        class), ..]``. Returns a pair of empty tuples if no classes were
        found.
        """
        if len(codewords) != len(codeword):
            raise ValueError("Lenth of `codewords` must be equal to `codeword` length")
        classes = []
        for class_, word in codewords.items():
            for i, bit in enumerate(word):
                if bit == on:
                    mse = (float(bit) - codeword[i]) ** 2
                    if mse <= error:
                        classes.append((mse, class_))
                    break
        return sorted(classes)

    def get_photos_with_class(self, session, metadata, filter_):
        """Return photos with corresponding class from the database.

        Photos obtained from the database are filtered by rules set in the
        `filter_` parameter. Returned rows are 2-tuples ``(photo_path,
        class)``.
        """
        if 'class' not in filter_:
            raise ValueError("The filter is missing the 'class' key")
        if isinstance(filter_, dict):
            filter_ = Struct(filter_)
        for key in vars(filter_):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in filter" % key)

        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = Base.classes.photos_taxa
        Taxa = Base.classes.taxa
        Rank = Base.classes.ranks

        # Construct the sub queries.
        stmt1 = session.query(PhotosTaxa.photo_id, Taxa.name.label('genus')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'genus').subquery()
        stmt2 = session.query(PhotosTaxa.photo_id, Taxa.name.label('section')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'section').subquery()
        stmt3 = session.query(PhotosTaxa.photo_id, Taxa.name.label('species')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'species').subquery()

        # Construct the query.
        q = session.query(Photos.path, getattr(filter_, 'class')).\
            join(stmt1).\
            outerjoin(stmt2).\
            join(stmt3)

        # Add the WHERE clauses to the query.
        if 'where' in filter_:
            for rank, taxon in vars(filter_.where).items():
                if rank == 'genus':
                    q = q.filter(stmt1.c.genus == taxon)
                elif rank == 'section':
                    q = q.filter(stmt2.c.section == taxon)
                elif rank == 'species':
                    q = q.filter(stmt3.c.species == taxon)

        return q

    def get_classes_from_filter(self, session, metadata, filter_):
        """Return the classes for a classification filter.

        Requires access to a database via an SQLAlchemy Session and
        MetaData object.

        This is a generator that returns one class at a time. The unique
        set of classes for the classification filter `filter_` are returned.
        """
        if 'class' not in filter_:
            raise ValueError("The filter is missing the 'class' key")
        if isinstance(filter_, dict):
            filter_ = Struct(filter_)
        for key in vars(filter_):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in filter" % key)

        # Poduce a set of mappings from the MetaData.
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = {'class': Base.classes.photos_taxa}
        Taxa = {'class': Base.classes.taxa}
        Ranks = {'class': Base.classes.ranks}

        # Construct the query, ORM style.
        q = session.query(Photos.id, Taxa['class'].name)

        if 'where' in filter_ and filter_.where:
            for rank, name in vars(filter_.where).items():
                PhotosTaxa[rank] = orm.aliased(Base.classes.photos_taxa)
                Taxa[rank] = orm.aliased(Base.classes.taxa)
                Ranks[rank] = orm.aliased(Base.classes.ranks)

                q = q.join(PhotosTaxa[rank], PhotosTaxa[rank].photo_id == Photos.id).\
                    join(Taxa[rank]).join(Ranks[rank]).\
                    filter(Ranks[rank].name == rank, Taxa[rank].name == name)

        # The classification column.
        rank = getattr(filter_, 'class')
        q = q.join(PhotosTaxa['class'], PhotosTaxa['class'].photo_id == Photos.id).\
            join(Taxa['class']).join(Ranks['class']).\
            filter(Ranks['class'].name == rank)

        # Order by classification.
        q = q.group_by(Taxa['class'].name)

        # Return the results.
        for (_, class_) in q:
            yield class_

    def get_taxon_hierarchy(self, session, metadata):
        """Return the taxanomic hierarchy for photos in the database.

        The hierarchy is returned as a dictionary in the format
        ``{genus: {section: [species, ..], ..}, ..}``.
        """
        hierarchy = {}

        for genus, section, species in  self.get_taxa(session, metadata):
            if genus not in hierarchy:
                hierarchy[genus] = {}
            if section not in hierarchy[genus]:
                hierarchy[genus][section] = []
            hierarchy[genus][section].append(species)
        return hierarchy

    def get_taxa(self, session, metadata):
        """Return the taxa from the database.

        Taxa are returned as (genus, section, species) tuples.
        """
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = Base.classes.photos_taxa
        Taxa = Base.classes.taxa
        Rank = Base.classes.ranks

        # Construct the sub queries.
        stmt1 = session.query(PhotosTaxa.photo_id, Taxa.name.label('genus')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'genus').subquery()
        stmt2 = session.query(PhotosTaxa.photo_id, Taxa.name.label('section')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'section').subquery()
        stmt3 = session.query(PhotosTaxa.photo_id, Taxa.name.label('species')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'species').subquery()

        # Construct the query.
        q = session.query(Photos.id, 'genus', 'section', 'species').\
            join(stmt1).\
            outerjoin(stmt2).\
            join(stmt3).\
            group_by('genus', 'section', 'species')

        for _, genus, section, species in q:
            yield (genus,section,species)

    def classification_hierarchy_filters(self, levels, hr, path=[]):
        """Return the classification filter for each path in a hierarchy.

        Returns the classification filter for each possible path in the
        hierarchy `hr`. The name of each level in the hierarchy must be set
        in the sequence `levels`. The sequence `path` holds the position in
        the hierarchy. Filters that return no classes are not returned.
        """
        filter_ = {}

        # The level number that is being classfied (0 based).
        level_no = len(path)

        if level_no > len(levels) - 1:
            raise ValueError("Maximum classification hierarchy depth exceeded")

        # Set the level to classify on.
        filter_['class'] = levels[level_no]

        # Set the where fields.
        filter_['where'] = {}
        for i, class_ in enumerate(path):
            name = levels[i]
            filter_['where'][name] = class_

        # Get the classes for the current hierarchy path.
        classes = self.get_childs_from_hierarchy(hr, path)

        # Only return the filter if the classes are set.
        if classes != [None]:
            yield filter_

        # Stop iteration if the last level was classified.
        if level_no == len(levels) - 1:
            raise StopIteration()

        # Recurse into lower hierarchy levels.
        for c in classes:
            for f in self.classification_hierarchy_filters(levels, hr,
                    path+[c]):
                yield f

    def get_childs_from_hierarchy(self, hr, path=[]):
        """Return the child node names for a node in a hierarchy.

        Returns a list of child node names of the hierarchy `hr` at node
        with the path `path`. The hierarchy `hr` is a nested dictionary,
        where bottom level nodes are lists. Which node to get the childs
        from is specified by `path`, which is a list of the node names up
        to that node. An empty list for `path` means the names of the nodes
        of the top level are returned.
        """
        nodes = hr.copy()
        try:
            for name in path:
                nodes = nodes[name]
        except:
            raise ValueError("No such path `%s` in the hierarchy" % '/'.join(path))

        if isinstance(nodes, dict):
            names = nodes.keys()
        elif isinstance(nodes, list):
            names = nodes
        else:
            raise ValueError("Incorrect hierarchy format")

        return names

    def readable_filter(self, filter_):
        """Return a human-readable description of a classification filter."""
        class_ = filter_.get('class')
        where = filter_.get('where', {})
        where_n = len(where)
        where_s = ""
        for i, (k,v) in enumerate(where.items()):
            if i > 0 and i < where_n - 1:
                where_s += ", "
            elif where_n > 1 and i == where_n - 1:
                where_s += " and "
            where_s += "%s is %s" % (k,v)

        if where_n > 0:
            return "%s where %s" % (class_, where_s)
        return "%s" % class_

class Phenotyper(object):
    """Generate numerical features from an image."""

    def __init__(self):
        self.path = None
        self.config = None
        self.img = None
        self.mask = None
        self.bin_mask = None

    def set_image(self, path):
        self.img = cv2.imread(path)
        if self.img == None or self.img.size == 0:
            raise IOError("Failed to read image %s" % path)

        self.path = path
        self.config = None
        self.mask = None
        self.bin_mask = None

        return self.img

    def set_config(self, config):
        """Set the YAML configurations object."""
        if not isinstance(config, Struct):
            raise TypeError("Configurations object must be of type Struct, not %s" % type(config))
        self.config = config

    def grabcut_with_margin(self, img, iters=5, margin=5):
        """Segment image into foreground and background pixels.

        Runs the GrabCut algorithm for segmentation. Returns an 8-bit
        single-channel mask. Its elements may have one of following values:
            * ``cv2.GC_BGD`` defines an obvious background pixel.
            * ``cv2.GC_FGD`` defines an obvious foreground pixel.
            * ``cv2.GC_PR_BGD`` defines a possible background pixel.
            * ``cv2.GC_PR_FGD`` defines a possible foreground pixel.

        The GrabCut algorithm is executed with `iters` iterations. The ROI is set
        to the entire image, with a margin of `margin` pixels from the edges.
        """
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdmodel = np.zeros((1,65), np.float64)
        fgdmodel = np.zeros((1,65), np.float64)
        rect = (margin, margin, img.shape[1]-margin*2, img.shape[0]-margin*2)
        cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, iters, cv2.GC_INIT_WITH_RECT)
        return mask

    def __preprocess(self):
        if self.img is None:
            raise RuntimeError("No image is loaded")

        if 'preprocess' not in self.config:
            return

        # Scale the image down if its perimeter exceeds the maximum (if set).
        perim = sum(self.img.shape[:2])
        max_perim = getattr(self.config.preprocess, 'maximum_perimeter', None)
        if max_perim and perim > max_perim:
            logging.info("Scaling down...")
            rf = float(max_perim) / perim
            self.img = cv2.resize(self.img, None, fx=rf, fy=rf)

        # Perform color enhancement.
        color_enhancement = getattr(self.config.preprocess, 'color_enhancement', None)
        if color_enhancement:
            for method, args in vars(color_enhancement).iteritems():
                if method == 'naik_murthy_linear':
                    logging.info("Color enhancement...")
                    self.img = ft.naik_murthy_linear(self.img)
                else:
                    raise ValueError("Unknown color enhancement method '%s'" % method)

        # Perform segmentation.
        try:
            segmentation = self.config.preprocess.segmentation.grabcut
        except:
            segmentation = {}

        if segmentation:
            logging.info("Segmenting...")
            iterations = getattr(segmentation, 'iterations', 5)
            margin = getattr(segmentation, 'margin', 1)
            output_folder = getattr(segmentation, 'output_folder', None)

            # Create a binary mask for the largest contour.
            self.mask = self.grabcut_with_margin(self.img, iterations, margin)
            self.bin_mask = np.where((self.mask==cv2.GC_FGD) + (self.mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
            contour = ft.get_largest_contour(self.bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contour == None:
                raise ValueError("No contour found for binary image")
            self.bin_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            cv2.drawContours(self.bin_mask, [contour], 0, 255, -1)

            # Save the masked image to the output folder.
            if output_folder and os.path.isdir(output_folder):
                img_masked = cv2.bitwise_and(self.img, self.img, mask=self.bin_mask)
                fname = os.path.basename(self.path)
                out_path = os.path.join(output_folder, fname)
                cv2.imwrite(out_path, img_masked)

    def make(self):
        if self.img == None:
            raise ValueError("No image loaded")

        #logging.info("Processing %s ..." % self.path)

        self.__preprocess()

        logging.info("Extracting features...")

        data_row = []

        if not 'features' in self.config:
            raise RuntimeError("Features to extract not set. Nothing to do.")

        for feature, args in vars(self.config.features).iteritems():
            if feature == 'color_histograms':
                logging.info("- Running color:histograms...")
                data = self.get_color_histograms(self.img, args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'color_bgr_means':
                logging.info("- Running color:bgr_means...")
                data = self.get_color_bgr_means(self.img, args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'shape_outline':
                logging.info("- Running shape:outline...")
                data = self.get_shape_outline(args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'shape_360':
                logging.info("- Running shape:360...")
                data = self.get_shape_360(args, self.bin_mask)
                data_row.extend(data)

            else:
                raise ValueError("Unknown feature '%s'" % feature)

        return data_row

    def get_color_histograms(self, src, args, bin_mask=None):
        histograms = []
        for colorspace, bins in vars(args).iteritems():
            if colorspace.lower() == "bgr":
                colorspace = ft.CS_BGR
                img = src
            elif colorspace.lower() == "hsv":
                colorspace = ft.CS_HSV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            elif colorspace.lower() == "luv":
                colorspace = ft.CS_LUV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2LUV)
            else:
                raise ValueError("Unknown colorspace '%s'" % colorspace)

            hists = ft.color_histograms(img, bins, bin_mask, colorspace)

            for hist in hists:
                hist = cv2.normalize(hist, None, -1, 1, cv2.NORM_MINMAX)
                histograms.extend( hist.ravel() )
        return histograms

    def get_color_bgr_means(self, src, args, bin_mask=None):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        # Get the contours from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Create a masked image.
        img = cv2.bitwise_and(src, src, mask=bin_mask)

        bins = getattr(args, 'bins', 20)
        output = ft.color_bgr_means(img, contour, bins)

        # Normalize data to range -1 .. 1
        return output * 2.0 / 255 - 1

    def get_shape_outline(self, args, bin_mask):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        k = getattr(args, 'k', 15)

        # Obtain contours (all points) from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Get the outline.
        outline = ft.shape_outline(contour, k)

        # Compute the delta's for the horizontal and vertical point pairs.
        shape = []
        for x, y in outline:
            delta_x = x[0] - x[1]
            delta_y = y[0] - y[1]
            shape.append(delta_x)
            shape.append(delta_y)

        # Normalize results.
        shape = np.array(shape, dtype=np.float32)
        shape = cv2.normalize(shape, None, -1, 1, cv2.NORM_MINMAX)

        return shape.ravel()

    def get_shape_360(self, args, bin_mask):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        rotation = getattr(args, 'rotation', 0)
        step = getattr(args, 'step', 1)
        t = getattr(args, 't', 8)
        output_functions = getattr(args, 'output_functions', {'mean_sd': True})

        # Get the largest contour from the binary mask.
        contour = ft.get_largest_contour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Set the rotation.
        if rotation == 'FIT_ELLIPSE':
            box = cv2.fitEllipse(contour)
            rotation = int(box[2])
        if not 0 <= rotation <= 179:
            raise ValueError("Rotation must be in the range 0 to 179, found %s" % rotation)

        # Extract shape feature.
        intersects, center = ft.shape_360(contour, rotation, step, t)

        # Create a masked image.
        if 'color_histograms' in output_functions:
            img_masked = cv2.bitwise_and(self.img, self.img, mask=bin_mask)

        # Run the output function for each angle.
        means = []
        sds = []
        histograms = []
        for angle in range(0, 360, step):
            for f_name, f_args in vars(output_functions).iteritems():
                # Mean distance + standard deviation.
                if f_name == 'mean_sd':
                    distances = []
                    for p in intersects[angle]:
                        d = ft.point_dist(center, p)
                        distances.append(d)

                    if len(distances) == 0:
                        mean = 0
                        sd = 0
                    else:
                        mean = np.mean(distances, dtype=np.float32)
                        if len(distances) > 1:
                            sd = np.std(distances, ddof=1, dtype=np.float32)
                        else:
                            sd = 0

                    means.append(mean)
                    sds.append(sd)

                # Color histograms.
                if f_name == 'color_histograms':
                    # Get a line from the center to the outer intersection point.
                    line = None
                    if len(intersects[angle]) > 0:
                        line = ft.extreme_points([center] + intersects[angle])

                    # Create a mask for the line, where the line is foreground.
                    line_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                    if line != None:
                        cv2.line(line_mask, tuple(line[0]), tuple(line[1]), 255, 1)

                    # Create histogram from masked + line masked image.
                    hists = self.get_color_histograms(img_masked, f_args, line_mask)
                    histograms.append(hists)

        # Normalize results.
        if 'mean_sd' in output_functions:
            means = cv2.normalize(np.array(means), None, -1, 1, cv2.NORM_MINMAX)
            sds = cv2.normalize(np.array(sds), None, -1, 1, cv2.NORM_MINMAX)

        # Group the means+sds together.
        means_sds = np.array(zip(means, sds)).flatten()

        return np.append(means_sds, histograms)

class TrainData(object):
    """Class for storing training data."""

    def __init__(self, num_input = 0, num_output = 0):
        self.num_input = num_input
        self.num_output = num_output
        self.labels = []
        self.input = []
        self.output = []
        self.counter = 0

    def read_from_file(self, path, dependent_prefix="OUT:"):
        """Reads training data from file.

        Data is loaded from TSV file `path`. File must have a header row,
        and columns with a name starting with `dependent_prefix` are used as
        classification columns. Optionally, sample labels can be stored in
        a column with name "ID". All remaining columns are used as predictors.
        """
        with open(path, 'r') as fh:
            reader = csv.reader(fh, delimiter="\t")

            # Figure out the format of the data.
            header = reader.next()
            input_start = None
            output_start = None
            label_idx = None
            for i, field in enumerate(header):
                if field == "ID":
                    label_idx = i
                elif field.startswith(dependent_prefix):
                    if output_start == None:
                        output_start = i
                    self.num_output += 1
                else:
                    if input_start == None:
                        input_start = i
                    self.num_input += 1

            if self.num_input == 0:
                raise IOError("No input columns found in training data.")
            if self.num_output  == 0:
                raise IOError("No output columns found in training data.")

            input_end = input_start + self.num_input
            output_end = output_start + self.num_output

            for row in reader:
                if label_idx != None:
                    self.labels.append(row[label_idx])
                else:
                    self.labels.append(None)
                self.input.append(row[input_start:input_end])
                self.output.append(row[output_start:output_end])

            self.finalize()

    def __len__(self):
        return len(self.input)

    def __iter__(self):
        return self

    def next(self):
        if self.counter >= len(self.input):
            self.counter = 0
            raise StopIteration
        else:
            self.counter += 1
            i = self.counter - 1
            return (self.labels[i], self.input[i], self.output[i])

    def append(self, input, output, label=None):
        if isinstance(self.input, np.ndarray):
            raise ValueError("Cannot add data once finalized")
        if len(input) != self.num_input:
            raise ValueError("Incorrect input array length (expected length of %d)" % self.num_input)
        if len(output) != self.num_output:
            raise ValueError("Incorrect output array length (expected length of %d)" % self.num_output)

        self.labels.append(label)
        self.input.append(input)
        self.output.append(output)

    def finalize(self):
        self.input = np.array(self.input).astype(float)
        self.output = np.array(self.output).astype(float)

    def normalize_input_columns(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this function")

        for col in range(self.num_input):
            tmp = cv2.normalize(self.input[:,col], None, alpha, beta, norm_type)
            self.input[:,col] = tmp[:,0]

    def normalize_input_rows(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this function")

        for i, row in enumerate(self.input):
            self.input[i] = cv2.normalize(row, None, alpha, beta, norm_type).reshape(-1)

    def round_input(self, decimals=4):
        self.input = np.around(self.input, decimals)

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

class TrainANN(object):
    """Train an artificial neural network."""

    def __init__(self):
        self.ann = None
        self.connection_rate = 1
        self.learning_rate = 0.7
        self.hidden_layers = 1
        self.hidden_neurons = 8
        self.epochs = 500000
        self.iterations_between_reports = self.epochs / 100
        self.desired_error = 0.0001
        self.training_algorithm = 'TRAIN_RPROP'
        self.activation_function_hidden = 'SIGMOID_STEPWISE'
        self.activation_function_output = 'SIGMOID_STEPWISE'
        self.train_data = None
        self.test_data = None

    def set_train_data(self, data):
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        self.train_data = data

    def set_test_data(self, data):
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        if data.num_input != self.train_data.num_input:
            raise ValueError("Number of inputs of test data must be same as train data")
        if data.num_output != self.train_data.num_output:
            raise ValueError("Number of output of test data must be same as train data")
        self.test_data = data

    def train(self, train_data):
        self.set_train_data(train_data)

        hidden_layers = [self.hidden_neurons] * self.hidden_layers
        layers = [self.train_data.num_input]
        layers.extend(hidden_layers)
        layers.append(self.train_data.num_output)

        sys.stderr.write("Network layout:\n")
        sys.stderr.write("* Neuron layers: %s\n" % layers)
        sys.stderr.write("* Connection rate: %s\n" % self.connection_rate)
        if self.training_algorithm not in ('FANN_TRAIN_RPROP',):
            sys.stderr.write("* Learning rate: %s\n" % self.learning_rate)
        sys.stderr.write("* Activation function for the hidden layers: %s\n" % self.activation_function_hidden)
        sys.stderr.write("* Activation function for the output layer: %s\n" % self.activation_function_output)
        sys.stderr.write("* Training algorithm: %s\n" % self.training_algorithm)

        self.ann = libfann.neural_net()
        self.ann.create_sparse_array(self.connection_rate, layers)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_activation_function_hidden(getattr(libfann, self.activation_function_hidden))
        self.ann.set_activation_function_output(getattr(libfann, self.activation_function_output))
        self.ann.set_training_algorithm(getattr(libfann, self.training_algorithm))

        fann_train_data = libfann.training_data()
        fann_train_data.set_train_data(self.train_data.get_input(), self.train_data.get_output())

        self.ann.train_on_data(fann_train_data, self.epochs, self.iterations_between_reports, self.desired_error)
        return self.ann

    def test(self, test_data):
        self.set_test_data(test_data)

        fann_test_data = libfann.training_data()
        fann_test_data.set_train_data(self.test_data.get_input(), self.test_data.get_output())

        self.ann.reset_MSE()
        self.ann.test_data(fann_test_data)

        return self.ann.get_MSE()