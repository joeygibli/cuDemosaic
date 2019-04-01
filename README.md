# cuDemosaic: Parallel Demosaic Algorithm for RAW Images

![Bayer Filter Bilinear Demosaic](https://github.com/joeygibli/cuDemosaic/blob/master/bayer_filter.png)

Converting RAW images to a format compatible with computer screens requires a process called demosaicing. Pixels on an image sensor only store exclusively red, green, or blue color levels. Therefore, when an image is presented on a computer screen certain RGB color levels must be infered. **Bilinear demosaicing**, one of the most common and simplest approaches, computes RGB values by taking the average of neighboring pixels in the surrounding area.

Specialized hardware is often needed to wildly efficient demosaic speeds required for real-time video. Our goal is achieve fast demosaic speeds by closely integrating standard hardware and with intelligent software speeds. For image smoothness, the human eye cannot detect flicker at about 70fps (or 14ms per frame). Using OpenMP and Intel Intrinsics, and clever design we are able to demosaic individual frames using a custom bilinear demosaic algorithm in about 13ms – beating the 70fps goal.

Below is a description of the algorithm:

![Legend](https://github.com/joeygibli/cuDemosaic/blob/master/legend.png)

We perform the demosaicing in parallel across ribbons of the RAW image. These columns are independent of each other in terms of writing to the final output image, therefore we avoid race conditions. Above is a legend for the visuals below. We demonstrate on a red/green Bayer filter row, but the algorithm extends to blue/green rows as well.

![Horizontal](https://github.com/joeygibli/cuDemosaic/blob/master/horizontal_interp.png)

On a red/green row, the horizontal interpolation will determine the green color levels of a red pixel and the red color levels of the green pixel. By reading from the Bayer filter RAW data and aligning the appropriate pixels we can add process 16 pixels in parallel using vectorize CPU instructions. Masked operations allow us to blend vectors and perform operations on only particular entries.

![Vertical](https://github.com/joeygibli/cuDemosaic/blob/master/vertical_interp.png)

The vertical interpolation of a blue/green row will determine the red color values of that row based on the surrounding pixels. Since the blue pixel's red value is the average the red pixels on it's diagonal this row is dependent on the interpolated red value of the green pixel directly above the blue one. This can be inferred from the horizontal interpolation above and below.
