#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trainer for the artificial neural networks.

The following classes are defined:

* Phenotyper: Create phenotypes from an image.

Tasks:

* data: Create a tab separated file with training data.
* ann: Train an artificial neural network.
* test-ann: Test the performance of an artificial neural network.
* classify: Classify an image using an artificial neural network.
"""

import argparse
from contextlib import contextmanager
import logging
import mimetypes
import os
import sys

import cv2
import numpy as np
from pyfann import libfann
import sqlalchemy
import sqlalchemy.orm as orm
from sqlalchemy.ext.automap import automap_base
import yaml

# Import the feature extraction library.
# https://github.com/naturalis/feature-extraction
import features as ft

def main():
    # Print debug messages if the -d flag is set for the Python interpreter.
    # Otherwise just show log messages of type INFO.
    if sys.flags.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Setup the argument parser.
    parser = argparse.ArgumentParser(description="Generate training data " \
        "and train artificial neural networks.")
    subparsers = parser.add_subparsers(help="Specify which task to start.")

    # Create an argument parser for sub-command 'data'.
    help_data = """Create a tab separated file with training data.
    Preprocessing steps and features to extract must be set in a YAML file.
    See trainer.yml for an example."""

    parser_data = subparsers.add_parser('data',
        help=help_data, description=help_data)
    parser_data.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a YAML file with feature extraction parameters.")
    parser_data.add_argument("--db", metavar="DB",
        help="Path to a database file with photo meta data. If omitted " \
        "this defaults to a file photos.db in the photo's directory.")
    parser_data.add_argument("--output", "-o", metavar="FILE", required=True,
        help="Output file name for training data. Any existing file with " \
        "same name will be overwritten.")
    parser_data.add_argument("basepath", metavar="PATH",
        help="Base directory where to look for photo's. The database file" \
        "with photo meta data will be used to find photo's in this directory.")

    # Create an argument parser for sub-command 'ann'.
    help_ann = """Train an artificial neural network. Optional training
    parameters can be set in a separate YAML file. See orchids.yml
    for an example file."""

    parser_ann = subparsers.add_parser('ann',
        help=help_ann, description=help_ann)
    parser_ann.add_argument("--test-data", metavar="FILE",
        help="Path to tab separated file with test data.")
    parser_ann.add_argument("--conf", metavar="FILE",
        help="Path to a YAML file with ANN training parameters.")
    parser_ann.add_argument("--epochs", metavar="N", type=float,
        help="Maximum number of epochs. Overwrites value in --conf.")
    parser_ann.add_argument("--error", metavar="N", type=float,
        help="Desired mean square error on training data. Overwrites value " \
        "in --conf.")
    parser_ann.add_argument("--output", "-o", metavar="FILE", required=True,
        help="Output file name for the artificial neural network. Any " \
        "existing file with same name will be overwritten.")
    parser_ann.add_argument("data", metavar="TRAIN_DATA",
        help="Path to tab separated file with training data.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = """Test an artificial neural network. If `--output` is
    set, then `--conf` must also be set. See orchids.yml for an example YAML
    file with class names."""

    parser_test_ann = subparsers.add_parser('test-ann',
        help=help_test_ann, description=help_test_ann)
    parser_test_ann.add_argument("--ann", metavar="FILE", required=True,
        help="A trained artificial neural network.")
    parser_test_ann.add_argument("--output", "-o", metavar="FILE",
        help="Output file name for the test results. Specifying this " \
        "option will output a table with the classification result for " \
        "each sample.")
    parser_test_ann.add_argument("--conf", metavar="FILE",
        help="Path to a YAML file with class names.")
    parser_test_ann.add_argument("--error", metavar="N", type=float,
        default=0.01,
        help="The maximum mean square error for classification. Default " \
        "is 0.01")
    parser_test_ann.add_argument("data", metavar="TEST_DATA",
        help="Path to tab separated file containing test data.")

    # Create an argument parser for sub-command 'classify'.
    help_classify = """Classify an image. See orchids.yml for an example YAML
    file with class names."""

    parser_classify = subparsers.add_parser('classify',
        help=help_classify, description=help_classify)
    parser_classify.add_argument("--ann", metavar="FILE", required=True,
        help="Path to a trained artificial neural network file.")
    parser_classify.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a YAML file with class names.")
    parser_classify.add_argument("--error", metavar="N", type=float,
        default=0.01,
        help="The maximum error for classification. Default is 0.01")
    parser_classify.add_argument("image", metavar="IMAGE",
        help="Path to image file to be classified.")

    # Parse arguments.
    args = parser.parse_args()

    if sys.argv[1] == 'data':
        if args.db is None:
            args.db = os.path.join(args.basepath, 'photos.db')
        try:
            train_data(args.basepath, args.conf, args.db, args.output)
        except ValueError as e:
            logging.error(e)

    elif sys.argv[1] == 'ann':
        train_ann(args.data, args.output, args.test_data, args.conf, args)
    elif sys.argv[1] == 'test-ann':
        if args.output and not args.conf:
            logging.error("Option `--conf` must be set when `--output` is set.")
            sys.exit()
        test_ann(args.ann, args.data, args.output, args.conf, args.error)
    elif sys.argv[1] == 'classify':
        classify(args.image, args.ann, args.conf, args.error)

    sys.exit()

@contextmanager
def session_scope(db_path):
    """Provide a transactional scope around a series of operations."""
    engine = sqlalchemy.create_engine('sqlite:///%s' % os.path.abspath(db_path),
        echo=sys.flags.debug)
    Session = orm.sessionmaker(bind=engine)
    session = Session()
    metadata = sqlalchemy.MetaData()
    metadata.reflect(bind=engine)
    try:
        yield (session, metadata)
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def train_data(base_path, conf_path, db_path, output_path):
    """Generate training data."""

    # Check if the paths exists.
    for path in (base_path, conf_path, db_path):
        if path is None:
            raise ValueError("Path cannot be None")
        if not os.path.exists(path):
            raise IOError("Cannot open %s (no such file or directory)" % path)

    # Open the configuration file.
    yml = open_yaml(conf_path)

    # Check if the path exists.
    if 'preprocess' in yml and 'segmentation' in yml.preprocess:
        path = getattr(yml.preprocess.segmentation, 'output_folder', None)
        if path and not os.path.exists(path):
            raise IOError("Cannot open %s (no such file or directory)" % path)

    with session_scope(db_path) as (session, metadata):
        # Query the database.
        if 'class_query' not in yml:
            raise ValueError("Classification query not set in the configuration file. Missing option 'class_query'.")

        q = make_class_query(session, metadata, yml.class_query)
        for path, class_ in q:
            print path, class_

    sys.exit()

    # Get list of image files and set the classes.
    images = {}
    classes = []
    for item in os.listdir(base_path):
        path = os.path.join(base_path, item)
        if os.path.isdir(path):
            classes.append(item)
            images[item] = get_image_files(path)

    # Make codeword for each class.
    codewords = get_codewords(classes, -1, 1)

    # Construct the header row.
    header_primer = ["ID"]
    header_data = []
    header_out = []

    for feature, args in vars(yml.features).iteritems():
        if feature == 'color_histograms':
            for colorspace, bins in vars(args).iteritems():
                for ch, n in enumerate(bins):
                    for i in range(1, n+1):
                        header_data.append("%s:%d" % (colorspace[ch], i))

        if feature == 'color_bgr_means':
            bins = getattr(args, 'bins', 20)
            for i in range(1, bins+1):
                for axis in ("HOR", "VER"):
                    for ch in "BGR":
                        header_data.append("BGR_MN:%d.%s.%s" % (i,axis,ch))

        if feature == 'shape_outline':
            n = getattr(args, 'k', 15)
            for i in range(1, n+1):
                header_data.append("OUTLINE:%d.X" % i)
                header_data.append("OUTLINE:%d.Y" % i)

        if feature == 'shape_360':
            step = getattr(args, 'step', 1)
            output_functions = getattr(args, 'output_functions', {'mean_sd': 1})
            for f_name, f_args in vars(output_functions).iteritems():
                if f_name == 'mean_sd':
                    for i in range(0, 360, step):
                        header_data.append("360:%d.MN" % i)
                        header_data.append("360:%d.SD" % i)

                if f_name == 'color_histograms':
                    for i in range(0, 360, step):
                        for cs, bins in vars(f_args).iteritems():
                            for j, color in enumerate(cs):
                                for k in range(1, bins[j]+1):
                                    header_data.append("360:%d.%s:%d" % (i,color,k))

    # Write classification columns.
    dependent_prefix = "OUT:"
    if 'data' in yml:
        dependent_prefix = getattr(yml.data, 'dependent_prefix', dependent_prefix)
    for i in range(1, len(classes)+1):
        header_out.append("%s%d" % (dependent_prefix, i))

    # Generate the training data.
    with open(output_path, 'w') as out_file:
        # Write the header row.
        out_file.write( "%s\n" % "\t".join(header_primer + header_data + header_out) )

        # Set the training data.
        training_data = common.TrainData(len(header_data), len(classes))
        fp = Phenotyper()
        failed = []
        for im_class, files in images.items():
            for im_path in files:
                if fp.open(im_path, yml) == None:
                    logging.warning("Failed to read %s. Skipping." % im_path)
                    failed.append(im_path)
                    continue

                try:
                    data = fp.make()
                except ValueError as e:
                    logging.error("Phenotyper failed: %s" % e)
                    logging.warning("Skipping.")
                    failed.append(im_path)
                    continue

                assert len(data) == len(header_data), "Data length mismatch"

                training_data.append(data, codewords[im_class], label=im_path)

        training_data.finalize()

        # Round all values.
        training_data.round_input(4)

        # Write data rows.
        for label, input_data, output_data in training_data:
            row = []
            row.append( label )
            row.extend( input_data.astype(str) )
            row.extend( output_data.astype(str) )
            out_file.write( "%s\n" % "\t".join(row) )

    logging.info("Training data written to %s" % output_path)

    # Print list of failed objects.
    if len(failed) > 0:
        logging.warning("Some files could not be processed:")
        for path in failed:
            logging.warning("- %s" % path)

def make_class_query(session, metadata, query):
    """Construct a query from the `class_query` parameter."""
    if 'class' not in query:
        raise ValueError("The query is missing the 'class' key")
    for key in vars(query):
        if key not in ('where', 'class'):
            raise ValueError("Unknown key '%s' in query" % key)

    # Poduce a set of mappings from the MetaData.
    Base = automap_base(metadata=metadata)
    Base.prepare()

    # Get the table classes.
    Photos = Base.classes.photos
    PhotosTaxa = {'class': Base.classes.photos_taxa}
    Taxa = {'class': Base.classes.taxa}
    Ranks = {'class': Base.classes.ranks}

    # Construct the query, ORM style.
    q = session.query(Photos.path, Taxa['class'].name)

    if 'where' in query:
        for rank, name in vars(query.where).items():
            PhotosTaxa[rank] = orm.aliased(Base.classes.photos_taxa)
            Taxa[rank] = orm.aliased(Base.classes.taxa)
            Ranks[rank] = orm.aliased(Base.classes.ranks)

            q = q.join(PhotosTaxa[rank], Photos.photos_taxa_collection).\
                join(Taxa[rank]).join(Ranks[rank]).\
                filter(Ranks[rank].name == rank, Taxa[rank].name == name)

    q = q.join(PhotosTaxa['class'], Photos.photos_taxa_collection).\
        join(Taxa['class']).join(Ranks['class']).\
        filter(Ranks['class'].name == getattr(query, 'class'))

    return q

def train_ann(train_data_path, output_path, test_data_path=None, conf_path=None, args=None):
    """Train an artificial neural network."""
    for path in (train_data_path, test_data_path, conf_path):
        if path and not os.path.exists(path):
            logging.error("Cannot open %s (no such file or directory)" % path)
            return 1

    # Instantiate the ANN trainer.
    ann_trainer = common.TrainANN()
    if conf_path:
        yml = open_yaml(conf_path)
        if not yml:
            return 1

        if 'ann' in yml:
            ann_trainer.connection_rate = getattr(yml.ann, 'connection_rate', 1)
            ann_trainer.hidden_layers = getattr(yml.ann, 'hidden_layers', 1)
            ann_trainer.hidden_neurons = getattr(yml.ann, 'hidden_neurons', 8)
            ann_trainer.learning_rate = getattr(yml.ann, 'learning_rate', 0.7)
            ann_trainer.epochs = getattr(yml.ann, 'epochs', 100000)
            ann_trainer.desired_error = getattr(yml.ann, 'error', 0.00001)
            ann_trainer.training_algorithm = getattr(yml.ann, 'training_algorithm', 'FANN_TRAIN_RPROP')
            ann_trainer.activation_function_hidden = getattr(yml.ann, 'activation_function_hidden', 'FANN_SIGMOID_STEPWISE')
            ann_trainer.activation_function_output = getattr(yml.ann, 'activation_function_output', 'FANN_SIGMOID_STEPWISE')

    # These arguments overwrite parameters in the YAML file.
    if args:
        if args.epochs != None:
            ann_trainer.epochs = args.epochs
        if args.error != None:
            ann_trainer.desired_error = args.error

    ann_trainer.iterations_between_reports = ann_trainer.epochs / 100

    # Get the prefix for the classification columns.
    dependent_prefix = "OUT:"
    if 'data' in yml:
        dependent_prefix = getattr(yml.data, 'dependent_prefix', dependent_prefix)

    train_data = common.TrainData()
    try:
        train_data.read_from_file(train_data_path, dependent_prefix)
    except ValueError as e:
        logging.error("Failed to process the training data: %s" % e)
        exit(1)

    # Train the ANN.
    ann = ann_trainer.train(train_data)
    ann.save(output_path)
    logging.info("Artificial neural network saved to %s" % output_path)

    # Test the ANN with training data.
    logging.info("Testing the neural network...")
    error = ann_trainer.test(train_data)
    logging.info("Mean Square Error on training data: %f" % error)

    if test_data_path:
        test_data = common.TrainData()
        test_data.read_from_file(test_data_path, dependent_prefix)
        error = ann_trainer.test(test_data)
        logging.info("Mean Square Error on test data: %f" % error)

def test_ann(ann_path, test_data_path, output_path=None, conf_path=None, error=0.01):
    """Test an artificial neural network."""
    for path in (ann_path, test_data_path, conf_path):
        if path and not os.path.exists(path):
            logging.error("Cannot open %s (no such file or directory)" % path)
            return 1

    if output_path and not conf_path:
        raise ValueError("Argument `conf_path` must be set when `output_path` is set")

    if conf_path:
        yml = open_yaml(conf_path)
        if not yml:
            return 1
        if 'classes' not in yml:
            logging.error("Classes are not set in the YAML file. Missing object 'classes'.")
            return 1

    # Get the prefix for the classification columns.
    dependent_prefix = "OUT:"
    if 'data' in yml:
        dependent_prefix = getattr(yml.data, 'dependent_prefix', dependent_prefix)

    ann = libfann.neural_net()
    ann.create_from_file(ann_path)

    test_data = common.TrainData()
    try:
        test_data.read_from_file(test_data_path, dependent_prefix)
    except ValueError as e:
        logging.error("Failed to process the test data: %s" % e)
        exit(1)

    logging.info("Testing the neural network...")
    fann_test_data = libfann.training_data()
    fann_test_data.set_train_data(test_data.get_input(), test_data.get_output())

    ann.test_data(fann_test_data)

    mse = ann.get_MSE()
    logging.info("Mean Square Error on test data: %f" % mse)

    if not output_path:
        return

    out_file = open(output_path, 'w')
    out_file.write( "%s\n" % "\t".join(['ID','Class','Classification','Match']) )

    # Get codeword for each class.
    codewords = get_codewords(yml.classes)

    total = 0
    correct = 0
    for label, input, output in test_data:
        total += 1
        row = []

        if label:
            row.append(label)
        else:
            row.append("")

        if len(codewords) != len(output):
            logging.error("Codeword length (%d) does not match the number of classes. "
                "Please make sure the correct classes are set in %s" % (len(output), conf_path))
            exit(1)

        class_e = get_classification(codewords, output, error)
        assert len(class_e) == 1, "The codeword for a class can only have one positive value"
        row.append(class_e[0])

        codeword = ann.run(input)
        try:
            class_f = get_classification(codewords, codeword, error)
        except ValueError as e:
            logging.error("Classification failed: %s" % e)
            return 1
        row.append(", ".join(class_f))

        # Check if the first items of the classifications match.
        if len(class_f) > 0 and class_f[0] == class_e[0]:
            row.append("+")
            correct += 1
        else:
            row.append("-")

        out_file.write( "%s\n" % "\t".join(row) )

    fraction = float(correct) / total
    out_file.write( "%s\n" % "\t".join(['','','',"%.3f" % fraction]) )
    out_file.close()

    logging.info("Correctly classified: %.1f%%" % (fraction*100))
    logging.info("Testing results written to %s" % output_path)

def classify(image_path, ann_path, conf_path, error):
    """Classify an image with a trained artificial neural network."""
    for path in (image_path, ann_path, conf_path):
        if path and not os.path.exists(path):
            logging.error("Cannot open %s (no such file or directory)" % path)
            return 1

    yml = open_yaml(conf_path)
    if not yml:
        return 1
    if 'classes' not in yml:
        logging.error("Classes are not set in the YAML file. Missing object 'classes'.")
        return 1

    # Load the ANN.
    ann = libfann.neural_net()
    ann.create_from_file(ann_path)

    # Get features from image.
    fp = Phenotyper()
    if fp.open(image_path, yml) == None:
        logging.error("Failed to read %s" % image_path)
        return 1
    features = fp.make()

    # Classify the image.
    codeword = ann.run(features)

    # Get codeword for each class.
    codewords = get_codewords(yml.classes)

    # Get the classification.
    try:
        classification = get_classification(codewords, codeword, error)
    except ValueError as e:
        logging.error("Failed to classify the image: %s" % e)
        return 1

    logging.info("Codeword: %s" % codeword)
    logging.info("Classification: %s" % ", ".join(classification))


def get_image_files(path):
    """Recursively obtain a list of image files from a path."""
    fl = []
    for item in os.listdir(path):
        im_path = os.path.join(path, item)
        if os.path.isdir(im_path):
            fl.extend( get_image_files(im_path) )
        elif os.path.isfile(im_path):
            mime = mimetypes.guess_type(im_path)[0]
            if mime and mime.startswith('image'):
                fl.append(im_path)
    return fl

def get_codewords(classes, neg=-1, pos=1):
    """Returns codewords for a list of classes."""
    n =  len(classes)
    codewords = {}
    for i, cls in enumerate(sorted(classes)):
        cw = [neg] * n
        cw[i] = pos
        codewords[cls] = cw
    return codewords

def get_classification(codewords, codeword, error=0.01):
    if len(codewords) != len(codeword):
        raise ValueError("Lenth of `codewords` must be equal to `codeword`")
    classes = []
    for cls, cw in codewords.items():
        for i, code in enumerate(cw):
            if code == 1.0 and (code - codeword[i])**2 < error:
                classes.append((codeword[i], cls))
    classes = [x[1] for x in sorted(classes, reverse=True)]
    return classes

def open_yaml(path):
    with open(path, 'r') as f:
        yml = yaml.load(f)
    yml = DictObject(yml)
    return yml

class DictObject(argparse.Namespace):
    def __init__(self, d):
        for a, b in d.iteritems():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [DictObject(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, DictObject(b) if isinstance(b, dict) else b)

class Phenotyper(object):
    """Generate numerical features from an image."""

    def __init__(self):
        self.path = None
        self.params = None
        self.img = None
        self.mask = None
        self.bin_mask = None

    def open(self, path, params):
        self.img = cv2.imread(path)
        self.params = params
        if self.img == None or self.img.size == 0:
            return None
        self.path = path
        self.mask = None
        self.bin_mask = None
        return self.img

    def _preprocess(self):
        if self.img == None:
            raise ValueError("No image loaded")

        if 'preprocess' not in self.params:
            return

        # Scale the image down if its perimeter exceeds the maximum (if set).
        perim = sum(self.img.shape[:2])
        max_perim = getattr(self.params.preprocess, 'maximum_perimeter', None)
        if max_perim and perim > max_perim:
            logging.info("Scaling down...")
            rf = float(max_perim) / perim
            self.img = cv2.resize(self.img, None, fx=rf, fy=rf)

        # Perform color enhancement.
        color_enhancement = getattr(self.params.preprocess, 'color_enhancement', None)
        if color_enhancement:
            for method, args in vars(color_enhancement).iteritems():
                if method == 'naik_murthy_linear':
                    logging.info("Color enhancement...")
                    self.img = ft.naik_murthy_linear(self.img)
                else:
                    raise ValueError("Unknown color enhancement method '%s'" % method)

        # Perform segmentation.
        segmentation = getattr(self.params.preprocess, 'segmentation', None)
        if segmentation:
            logging.info("Segmenting...")
            iterations = getattr(segmentation, 'iterations', 5)
            margin = getattr(segmentation, 'margin', 1)
            output_folder = getattr(segmentation, 'output_folder', None)

            # Create a binary mask for the largest contour.
            self.mask = ft.segment(self.img, iterations, margin)
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

        logging.info("Processing %s ..." % self.path)

        self._preprocess()

        logging.info("Extracting features...")

        data_row = []

        if not 'features' in self.params:
            raise ValueError("Features to extract not set. Nothing to do.")

        for feature, args in vars(self.params.features).iteritems():
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

if __name__ == "__main__":
    main()
