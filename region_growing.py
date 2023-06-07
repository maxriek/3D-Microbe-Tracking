import numpy as np


class RegionGrowing:
    def __init__(self, threshold, radius, rejection_threshold, n_values_to_ignore, progress_callback=None):
        self.threshold = threshold
        self.rejection_threshold = rejection_threshold
        self.radius = radius
        self.n_values_to_ignore = n_values_to_ignore
        self.progress_callback = progress_callback

    # get the coordinates of neighbour pixels for a given pixel in accordance to the given radius
    def get_all_neighbours(self, pixel, radius, max_coordinates):
        neighbours_coordinates_list = []
        current_x = pixel[0]
        current_y = pixel[1]
        y_max = max_coordinates[0] - 1
        x_max = max_coordinates[1] - 1

        # add all possible indices to the list neighboursIndices
        for i in range(current_x - radius, current_x + radius + 1):
            for n in range(current_y - radius, current_y + radius + 1):
                # filter all invalid indices - those are
                #  - indices with negative values
                #  - indices with values exceeding the total length (minus one)
                #    on the corresponding axis
                #  - the index of the current pixel itself
                if (i == current_x and n == current_y) or i < 0 or n < 0 or i > x_max or n > y_max:
                    continue
                neighbours_coordinates_list.append([i, n])

        return neighbours_coordinates_list

    # distance function: determine if a neighbour pixel is similar to a given pixel
    # load corresponding timestamps (for example npy file)->use as intensity image.
    # Compare values of pixels. The difference should not be greater than a given
    # threshold
    def is_in_region(self, current_pixel, neighbour, mhi, potential_background_values):
        current_pixel_intensity = mhi[current_pixel[1], current_pixel[0]]
        neighbour_pixel_intensity = mhi[neighbour[1], neighbour[0]]
        # ignore potential background values
        # filter out the most prominent n values in the mhi to enhance background elimination
        if current_pixel_intensity in potential_background_values:
            return False

        vec_time_diff = current_pixel_intensity * 1. - neighbour_pixel_intensity * 1.

        return abs(vec_time_diff) <= self.threshold

    # use histogram to filter out the most prominent n values in the mhi
    # to enhance background elimination
    def get_potential_background_values(self, mhi):
        max_pixel_value = np.max(mhi)
        min_pixel_value = np.min(mhi)
        counts, _ = np.histogram(mhi.ravel(), max_pixel_value, [min_pixel_value, max_pixel_value])
        largest_n_indices = np.argsort(counts)
        potential_background_values = largest_n_indices[-self.n_values_to_ignore:] + 1
        return potential_background_values

    # Implement Region Growing Algorithm: Starting with a given seed point, the algorithm
    # checks all the neighbour within a given radius and determine according to the
    # distance function if the pixels belong to the region or not. Pixels assigned to the region,
    # are processed recursively.
    def region_growing(self, mhi, seed):
        list_of_region_pixels = [seed]
        track_img = np.zeros_like(mhi, dtype=np.uint8)
        track_img[seed[1], seed[0]] = 255
        processed_pixels_matrix = np.zeros((mhi.shape[0], mhi.shape[1]))
        processed_pixels_matrix[seed[1], seed[0]] = 1

        potential_background_values = []
        if self.n_values_to_ignore is not None:
            potential_background_values = self.get_potential_background_values(mhi)

        iterations = 0
        while len(list_of_region_pixels):
            current_pixel = list_of_region_pixels[0]
            track_img[current_pixel[1], current_pixel[0]] = 255
            all_neighbours = self.get_all_neighbours(current_pixel, self.radius, mhi.shape)

            for neighbour in all_neighbours:
                if processed_pixels_matrix[neighbour[1], neighbour[0]] < self.rejection_threshold:
                    processed_pixels_matrix[neighbour[1], neighbour[0]] += 1
                    if self.is_in_region(current_pixel, neighbour, mhi, potential_background_values):
                        processed_pixels_matrix[neighbour[1], neighbour[0]] += self.rejection_threshold
                        list_of_region_pixels.append(neighbour)

            list_of_region_pixels.pop(0)
            iterations += 1
            self.show_progress(iterations, track_img)

        non_zero_track_values = np.nonzero(track_img)
        if len(non_zero_track_values[0]) < 100:
            print(f'#: no tracks found or track is to short (less than 100 pixels) - track length: {len(non_zero_track_values[0])}')

        return track_img

    def show_progress(self, iterations, track_img):
        if iterations % 1000 == 0:
            print(f'interation: {iterations} ## ')
            if self.progress_callback is not None:
                self.progress_callback(track_img)
