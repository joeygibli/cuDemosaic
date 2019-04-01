# cuDemosaic: Parallel Demosaic Algorithm for RAW Images

![Bayer Filter Bilinear Demosaic](https://github.com/joeygibli/cuDemosaic/blob/master/bayer_filter.png)

Converting RAW images to a format compatible with computer screens requires a process called demosaicing. Pixels on an image sensor only store exclusively red, green, or blue color levels. Therefore, when an image is presented on a computer screen certain RGB color levels must be infered. **Bilinear demosaicing**, one of the most common and simplest approaches, computes RGB values by taking the average of neighboring pixels in the surrounding area.

Specialized hardware is often needed to wildly efficient demosaic speeds required for real-time video. Our goal is achieve fast demosaic speeds by closely integrating standard hardware and with intelligent software speeds. For image smoothness, the human eye cannot detect flicker at about 70fps (or 14ms per frame). Using OpenMP and Intel Intrinsics, and clever we are able to demosaic individual frames using a custom bilinear demosaic algorithm in about 13ms – beating the 70fps goal.

![Legend](https://github.com/joeygibli/cuDemosaic/blob/master/legend.png)

![Horizontal](https://github.com/joeygibli/cuDemosaic/blob/master/horizontal_interp.png)
![Vertical](https://github.com/joeygibli/cuDemosaic/blob/master/vertical_interp.png)
