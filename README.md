# TaLiSSman: TrAnsmitted LIght Stack SegMentAtioN
Segmentation of bacteria growing in agar-pads, imaged by de-focused transmitted light stacks

## How it works
- Expected input images are stacks of 2D images, with Z-axis last : Image = [batch, Y, X, Z]. We advise stacks of 5 slices with a 0.2µm step, in range [-0.6µm, -1.4µm] (relatively to the focal plane)
- Segmentation is performed by regression of the Euclidean Distance Map (EDM).
- This repository does not include the downstream watershed step to obtain labeled images.

| Input Transmitted-Light Stack | Predicted EDM | Segmented Bacteria |
| :---:         |          :---: |          :---: |
| <img src="../../wiki/resources/inputStackREV.gif" width="300"> | <img src="../../wiki/resources/edm.png" width="300">    | <img src="../../wiki/resources/outputStackREV.gif" width="300"> |

Samples provided by Daniel Thédié, <a href="http://www.elkarouilab.fr/">El Karoui Lab</a>, University of Edinburg

## Network architecture:
- Based on U-net
- Before first layer a block is added to reduced Z-axis using 3D convolutions and 3D Z-maxpooling

More details can be found on my [website](https://www.sabilab.fr/project/2021/12/21/talissman.html)

## How to use it
Check the [wiki](../../wiki)! 
