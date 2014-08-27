import os

import nbclassify as nbc

from django.http import HttpResponseRedirect, Http404
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response, get_object_or_404
from django.core.context_processors import csrf
from django.conf import settings
from orchid.forms import UploadPictureForm
from orchid.models import Photo, Identity
from orchid.classify import ImageClassifier, open_yaml

ORCHID_CONF = os.path.join(settings.BASE_DIR, 'orchid', 'orchid.yml')
TAXA_DB = os.path.join(settings.BASE_DIR, 'orchid', 'taxa.db')
ANN_DIR = os.path.join(settings.BASE_DIR, 'orchid', 'orchid.ann')

def home(request):
    """Display the home page."""
    data = {}
    data.update(csrf(request))

    if request.method == 'POST':
        form = UploadPictureForm(request.POST, request.FILES)

        if form.is_valid():
            photo = form.save()

            # Keep track of which photo belongs to which session.
            try:
                request.session['photos'] += [photo.id]
            except:
                request.session['photos'] = [photo.id]

            data['photo'] = photo
            return render_to_response("orchid/photo.html", data)
        else:
            data['error_message'] = "Please select a valid image file."

    data['form'] = UploadPictureForm()
    return render_to_response("orchid/home.html", data)

def identify_ajax(request, photo_id):
    """Classify photo with ID `photo_id`.

    Each classification is saved as an Identity object in the database.
    Return the identities as an HTML table.
    """
    photo = get_object_or_404(Photo, pk=photo_id)
    ids = photo.identity_set.all()

    if not ids:
        if request.method != 'POST':
            raise Http404

        # Get the region of interest if it was set.
        roi = request.POST['roi']
        if roi:
            # Set the ROI for the photo.
            photo.roi = roi
            photo.save()

            # Convert to integer list.
            roi = roi.split(',')
            roi = [int(x) for x in roi]
        else:
            roi = None

        # Classify the photo.
        config = open_yaml(ORCHID_CONF)
        classifier = ImageClassifier(config, TAXA_DB)
        classifier.set_roi(roi)
        classes = classify_image(classifier, photo.photo.path, ANN_DIR)

        # Identify this photo.
        for c in classes:
            if not c.get('genus'):
                continue

            id_ = Identity(
                photo=photo,
                genus=c.get('genus'),
                section=c.get('section'),
                species=c.get('species'),
                error=c.get('error')
            )

            # Save the identity into the database.
            id_.save()

    ids = photo.identity_set.all()
    data = {'identities': ids}
    return render_to_response("orchid/result_ajax.html", data)

def classify_image(classifier, image_path, ann_dir):
    """Classify an image using a classfication hierarchy,

    Arguments are an instance of ImageClassifier `classifier`, file path to
    the image file `image_path`, and the path to the directory containing the
    artificial neural networks `ann_dir` for the specified classfication
    hierarchy set in `classifier`.

    Returns the classfications as a list of dictionaries, where each
    dictionary maps each level to the corresponding classfication. An
    additional key ``error`` specifies the mean square error for the entire
    classfication.
    """
    classes, errors = classifier.classify_with_hierarchy(image_path, ann_dir)

    # Check for failed classification.
    if not classes[0]:
        return []

    # Calculate the mean square error for each classification path.
    errors_classes = [(sum(e)/len(e),c) for e,c in zip(errors, classes)]

    # Get the level names.
    levels = classifier.get_classification_hierarchy_levels()

    # Create a list of all classifications.
    classes = []
    for error, classes_ in sorted(errors_classes):
        class_dict = {'error': error}
        for level, class_ in zip(levels, classes_):
            class_dict[level] = class_
        classes.append(class_dict)

    return classes

def result(request, photo_id):
    """Display the classification result for a photo with ID `photo_id`."""
    photo = Photo.objects.get(pk=photo_id)

    data = {}
    data.update(csrf(request))
    data['photo'] = photo
    data['roi'] = None

    # Get the ROI.
    if photo.roi:
        y,y2,x,x2 = photo.roi.split(",")
        data['roi'] = {'y':y, 'y2':y2, 'x':x, 'x2':x2}

    # Get the identities from the database.
    data['identities'] = photo.identity_set.all()

    return render_to_response("orchid/result.html", data)

def my_photos(request):
    """Display the photos that were identified in a session."""
    data = {}
    photos = []

    try:
        pks = request.session['photos']
    except:
        pks = []

    # Get the photos that belong to this session.
    for photo_id in sorted(pks, reverse=True):
        try:
            photo = Photo.objects.get(pk=photo_id)
            photos.append(photo)
        except (KeyError, Photo.DoesNotExist):
            # We can't modify session values directly.
            ids = request.session['photos']
            ids.remove(photo_id)
            request.session['photos'] = ids

    data['photos'] = photos
    return render_to_response("orchid/my_photos.html", data)
