# MIT License
# Copyright (c) 2023 Barthélemy Paléologue
# Source: https://github.com/BarthPaleologue/ALPR
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense,

import cv2 as cv
import numpy as np


def kmean_segmentation(img, n_iterations=5, n_classes=10):
    # Initial classes centers
    max_val = np.max(img)
    min_val = np.min(img)
    interval = float(max_val - min_val) / n_classes
    classes_center = np.array(
        [interval / 2 + min_val + interval * i for i in range(n_classes)]
    )

    classification = np.zeros(img.shape)
    classes_pixel_values = []
    for i in range(n_iterations):
        # Update pixel classes
        classes_pixel_values = [[] for i in range(n_classes)]
        for l in range(0, img.shape[0]):
            for c in range(0, img.shape[1]):
                dist_to_classes = np.abs(classes_center - img[l, c])
                class_index = np.argmin(dist_to_classes)
                classes_pixel_values[class_index].append(img[l, c])
                classification[l, c] = class_index

        # Update class centers
        for i in range(n_classes):
            center = np.mean(np.array(classes_pixel_values[i]))
            classes_center[i] = center

    return classification, classes_center


def platePerspectiveUnwarpingWithWhite(
    imageRGB, plate_xmin, plate_ymin, plate_xmax, plate_ymax
):
    image = np.copy(imageRGB)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)
    padding = 0
    xmin = plate_xmin - padding
    ymin = plate_ymin - padding
    xmax = plate_xmax + padding
    ymax = plate_ymax + padding
    image_gray = image_gray[ymin:ymax, xmin:xmax]
    image = image[ymin:ymax, xmin:xmax]
    # K-means
    classes_count = 3
    seg, _ = kmean_segmentation(image_gray, 20, classes_count)
    seg = np.uint8(seg)
    # Extract the class closest to white (The last class)
    classes_dist = []
    img_center = np.array([seg.shape[0] / 2, seg.shape[1] / 2])
    for i in range(1, classes_count):
        pixels_to_center = np.transpose(np.nonzero(seg == i)) - img_center
        pixels_to_center_dist = np.linalg.norm(pixels_to_center, axis=1) / np.sqrt(
            pow(seg.shape[0], 2) + pow(seg.shape[1], 2)
        )
        mean_dist = np.mean(pixels_to_center_dist)
        size = np.count_nonzero(seg == i) / (seg.shape[0] * seg.shape[1])
        classes_dist.append(mean_dist / size)
    # Get the class close to the center
    seg_class = np.argmin(np.array([d for i, d in enumerate(classes_dist)]))
    class0 = np.uint8(1 * (seg == seg_class + 1))
    # Biggest connected component
    ret, comps = cv.connectedComponents(class0)
    colors, counts = np.unique(comps.reshape(-1), return_counts=True, axis=0)
    colors = np.delete(colors, 0)
    counts = np.delete(counts, 0)
    compMax = comps == colors[np.argmax(counts)]
    compMax = np.uint8(compMax)
    # We extract the contours of the component
    contours, _ = cv.findContours(
        compMax,
        mode=cv.RETR_EXTERNAL,  # We should get a single contours for the single connected component
        method=cv.CHAIN_APPROX_SIMPLE,
    )
    # We get the oriented bounding box
    rect = cv.minAreaRect(contours[0])
    corners = cv.boxPoints(rect)
    Aind = np.argmin(np.linalg.norm(corners - np.zeros(corners.shape), axis=1))
    corners = np.array([corners[(i + Aind) % 4] for i in range(4)])
    # We apply perspective correction
    out_height = image.shape[0]
    out_width = image.shape[1]
    output_size = (out_width, out_height)
    perspective_matrix = cv.getPerspectiveTransform(
        corners,
        np.array(
            [[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]],
            dtype=np.float32,
        ),
    )
    corrected_image = cv.warpPerspective(
        image, perspective_matrix, output_size, cv.WARP_INVERSE_MAP
    )
    return corrected_image


def platePerspectiveUnwarpingWithSuperPixel(
    imageRGB,
    plate_xmin,
    plate_ymin,
    plate_xmax,
    plate_ymax,
    plate_padding=0,
    main_color=np.array([200, 200, 200]),
    color_dist=150,
):
    image = np.copy(imageRGB)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)
    xmin = plate_xmin - plate_padding
    ymin = plate_ymin - plate_padding
    xmax = plate_xmax + plate_padding
    ymax = plate_ymax + plate_padding
    image_gray = image_gray[ymin:ymax, xmin:xmax]
    image_src = image[ymin:ymax, xmin:xmax]

    image_gray = np.uint8(image_gray)
    edges = cv.Canny(image_gray, 50, 200)

    # find the pixels away from the edges
    edges_inv = np.bitwise_not(edges)
    dist_transform = cv.distanceTransform(edges_inv, cv.DIST_L2, 5)
    dist_transform = np.uint8(dist_transform)
    dist_most = cv.adaptiveThreshold(
        dist_transform,
        maxValue=dist_transform.max(),
        adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv.THRESH_BINARY,
        blockSize=11,
        C=0,
    )

    # Get their connected components
    dist_most = np.uint8(dist_most)
    ret, markers = cv.connectedComponents(dist_most)
    markers = markers + 1
    markers[dist_most == 0] = 0
    # Apply watershed
    markers = cv.watershed(image_src, markers)
    img_seg = np.copy(image_src)
    img_seg[markers == -1] = [255, 0, 0]
    # Choose superpixels based on color
    imgg = np.zeros(image_gray.shape)
    for i in range(ret):
        c = np.mean(img_seg[markers == i + 1], axis=0)  # Average color
        pixelCenter = np.mean(np.argwhere(markers == i + 1), axis=0)
        d = np.linalg.norm(main_color - c)
        if d < color_dist and (
            pixelCenter[0] > plate_padding
            and pixelCenter[0] < image_gray.shape[0] - plate_padding
            and pixelCenter[1] > plate_padding
            and pixelCenter[1] < image_gray.shape[1] - plate_padding
        ):
            imgg[markers == i + 1] = 255

    # Dilate selected super pixels to remove boundaries
    kernel = np.ones((2, 2), np.uint8)
    imgg = cv.dilate(imgg, kernel, iterations=1)
    imgg = np.uint8(imgg)

    # Detect contours
    imgg = np.uint8(imgg)
    contours, _ = cv.findContours(
        imgg,
        mode=cv.RETR_EXTERNAL,
        # mode=cv.RETR_LIST,
        method=cv.CHAIN_APPROX_SIMPLE,
    )
    if len(contours) == 0:
        return np.zeros(image_gray.shape)
    # Find best contour
    best_contour = contours[0]
    best_contour_weight = np.inf
    total_area = image_gray.shape[0] * image_gray.shape[1]
    for contour in contours:
        epsilon = np.min([image_gray.shape[0], image_gray.shape[1]]) * 0.1
        approx = cv.approxPolyDP(contour, epsilon, True)
        conv = cv.convexHull(contour)
        box = cv.minAreaRect(best_contour)
        box = cv.boxPoints(box)

        contourArea = cv.contourArea(contour)
        approxArea = cv.contourArea(approx)
        convArea = cv.contourArea(conv)
        boxArea = cv.contourArea(box)
        if contourArea > 0:
            weight0 = approxArea / contourArea
            weight1 = convArea / contourArea
            weight2 = convArea / total_area
        else:
            weight0 = 0
            weight1 = 0
            weight2 = 0

        if weight0 > 0 and weight1 > 0 and weight2 > 0:
            weight = abs(weight0 - weight1) / (weight2)
        else:
            weight = np.inf

        cc = cv.approxPolyDP(conv, 0.04 * cv.arcLength(conv, closed=True), True)
        if weight < best_contour_weight:
            best_contour_weight = weight
            best_contour = cc
    # Get box corners
    rect = cv.minAreaRect(best_contour)
    corners = cv.boxPoints(rect)
    # change corners order
    Aind = np.argmin(np.linalg.norm(corners - np.zeros(corners.shape), axis=1))
    corners = np.array([corners[(i + Aind) % 4] for i in range(4)])
    # Apply warping
    out_height = image_src.shape[0]
    out_width = image_src.shape[1]
    output_size = (out_width, out_height)
    perspective_matrix = cv.getPerspectiveTransform(
        corners,
        np.array(
            [[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]],
            dtype=np.float32,
        ),
    )
    corrected_image = cv.warpPerspective(
        image_src, perspective_matrix, output_size, cv.WARP_INVERSE_MAP
    )
    return corrected_image


def platePerspectiveUnwarping(
    imageRGB, plate_xmin, plate_ymin, plate_xmax, plate_ymax, plate_padding=0
):
    # Extract plate
    image = np.copy(imageRGB)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)
    xmin = plate_xmin - plate_padding
    ymin = plate_ymin - plate_padding
    xmax = plate_xmax + plate_padding
    ymax = plate_ymax + plate_padding
    image_gray = image_gray[ymin:ymax, xmin:xmax]
    image = image[ymin:ymax, xmin:xmax]

    # Edges detection
    image_gray = np.uint8(image_gray)
    edges = cv.Canny(image_gray, 50, 200)
    # Contours
    edges = np.uint8(edges)
    contours, _ = cv.findContours(
        edges, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
    )
    # Find best contour
    best_contour = contours[0]
    best_contour_weight = np.inf
    total_area = image_gray.shape[0] * image_gray.shape[1]
    for contour in contours:
        # Contour approximation
        epsilon = np.min([image_gray.shape[0], image_gray.shape[1]]) * 0.1
        approx = cv.approxPolyDP(contour, epsilon, True)
        # Convex hull
        conv = cv.convexHull(contour)
        # OBB
        box = cv.minAreaRect(best_contour)
        box = cv.boxPoints(box)

        contourArea = cv.contourArea(contour)
        approxArea = cv.contourArea(approx)
        convArea = cv.contourArea(conv)
        if contourArea > 0:
            weight0 = approxArea / contourArea
            weight1 = convArea / contourArea
            weight2 = convArea / total_area
        else:
            weight0 = 0
            weight1 = 0
            weight2 = 0

        if weight0 > 0 and weight1 > 0 and weight2 > 0:
            weight = abs(weight0 - weight1) / (weight2)
        else:
            weight = np.inf

        cc = cv.approxPolyDP(conv, 0.04 * cv.arcLength(conv, closed=True), True)
        if weight < best_contour_weight:
            best_contour_weight = weight
            best_contour = cc

    # Extract corners
    rect = cv.minAreaRect(best_contour)
    corners = cv.boxPoints(rect)
    Aind = np.argmin(np.linalg.norm(corners - np.zeros(corners.shape), axis=1))
    # Reorder
    corners = np.array([corners[(i + Aind) % 4] for i in range(4)])

    # Apply correction
    out_height = image_gray.shape[0]
    out_width = image_gray.shape[1]
    output_size = (out_width, out_height)
    perspective_matrix = cv.getPerspectiveTransform(
        corners,
        np.array(
            [[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]],
            dtype=np.float32,
        ),
    )
    corrected_image = cv.warpPerspective(
        image, perspective_matrix, output_size, cv.WARP_INVERSE_MAP
    )
    return corrected_image
