_this page is under construction_

# Multiclass pixel classifier
Deep learning segmentation method with low annotation requirement.

Similar to [Ilastik pixel classification](https://www.ilastik.org/documentation/pixelclassification/pixelclassification) procedure.

## Training Set
For cell segmentation, one typically defines 3 categories of pixels: inside cells, outside cells and cell outer edges (third category allows a better separation of neighboring cells).

### Generate a training set using BACMMAN software:
- [Install BACMMAN](https://github.com/jeanollion/bacmman/wiki/Installation)
- create a new dataset from the online library, `username = jeanollion`, `folder = Seg3Class`, `configuration = TrainingSet`. For help see the [doc](https://github.com/jeanollion/bacmman/wiki/Online-Configuration-Library#create-dataset-from-the-library).
- import images:
  - from the configuration tab set the import method that correspond to your images;
  - run the command `import/re-link images` from the menu `Run` : position should appear in the position list
- generate the training set by drawing areas of `Background` and `Contour` object classes
  - from the `data browsing` tab right click on a position and choose `open hyperstack`
  - choose the object class to edit by pression `i`
  - use the selection brush tool to draw areas for each object class. For contours should be closed, and with a linewidth of typically 3 pixels (double click on the tool to set the size). Ssee the [doc on manual edition](https://github.com/jeanollion/bacmman/wiki/Data-Curation#creation-1)
  - automatically close the contours: from the `home` tab select the position that have been edited, chose `Segment and Track` in the task panel and `Filled Contours` in the object panel. from the `Run` menu choose `Run selected Tasks`.
- export the training set:
  - from the `Home` tab select all the positions in the `Positions` panel
  - right-click on the `Task` panel and choose `Extract Dataset` > `Add new PixMClass dataset extraction task`, when prompted choose a directory and enter a filename ending in .h5 (or selected an existing file, in that case the dataset will be appended)
  - right-click on the `Task` panel and choose `Run all tasks`

## Model Training

## Prediction
