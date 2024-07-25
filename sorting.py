import math
import os
import random
import numpy as np
import pandas as pd
from functools import partial
import time

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.animation import FuncAnimation, PillowWriter

# TODO: This class should go in a utils.py file, since it can be used very widely
class AnimateBarPlot:
    @staticmethod
    def __animate_single_plot(frame, fig, ax, all_l: np.array, all_colors: list, max_height: float, title: str=""):
        l = all_l[frame]
        colors = all_colors[frame]

        ax.clear()

        ax.set_title(title)
        ax.set_ylim(0, max_height)
        ax.set_ylabel("Height")
        ax.set_xlabel("Index")

        ax.bar(range(len(l)), l, color=colors)
        fig.canvas.draw()

    def __make_gif_from_plots(self, all_l: np.array, all_colors: list, max_height: float, title: str="", save_fig=False):

        fig, ax = plt.subplots()
        plot_animation = FuncAnimation(fig,
                                       partial(self.__animate_single_plot,
                                               fig=fig,
                                               ax=ax,
                                               all_l=all_l,
                                               all_colors=all_colors,
                                               max_height=max_height,
                                               title=title),
                                       frames=len(all_l),
                                       interval=50,
                                       repeat=False)

        if save_fig:
            print("Saving figure")
            plot_animation.save(f'{title}_animation.gif', writer=PillowWriter(fps=5))
        else:
            plt.show()
        plt.close()

    def animate(self, all_l: np.array, all_colors: list, max_height: float, title: str="", save_fig: bool=False):
        self.__make_gif_from_plots(all_l, all_colors, max_height, title, save_fig)

class MakeSortingAlgorithmGIFs():
    @staticmethod
    def __insertion_sort_with_colors(l: list, base_colors: list) -> (list, list):

        all_l = [list(l)]
        all_colors = [list(base_colors[l])]

        length_l = len(l)

        # done_idx denotes the index we need to start consideration from. Insert sort works such that after a while,
        # some of the smallest numbers/leftmost indices will be sorted, so we don't need to consider those anymore.
        # Everything before done_idx is confirmed to be fully sorted.

        for done_idx in range(length_l-1):

            # The inner while loop takes the next number to the right and sorts it in its final place by iteratively checking if the number is less than the number to the left
            # When the inner while loop breaks, that means the next number has been sorted in its final place
            starting_idx = done_idx+1
            while l[starting_idx] < l[starting_idx-1] and starting_idx > 0:
                l[starting_idx-1:starting_idx+1] = [l[starting_idx], l[starting_idx-1]]

                # We've checked if the number at starting_idx is less than the one to the left and moved it left if so.
                # We now consider that number in its new place after switching and keep checking if it's less than the number to the left and switching its place if so,
                # until the number is sorted in its final place.
                starting_idx -= 1

                all_l.append(list(l))
                all_colors.append(list(base_colors[l]))

        return np.array(all_l), all_colors

    @staticmethod
    def __selection_sort_with_colors(l: list, base_colors: list) -> (list, list):
        sorted_up_to_idx = 0
        length = len(l)

        all_l = [list(l)]
        all_colors = [list(base_colors[l])]

        while sorted_up_to_idx < length:
            min_element = l[sorted_up_to_idx]
            min_idx = sorted_up_to_idx
            for i in range(sorted_up_to_idx, length):
                if l[i] < min_element:
                    min_element = l[i]
                    min_idx = i

            l[min_idx] = l[sorted_up_to_idx]
            l[sorted_up_to_idx] = min_element

            all_l.append(list(l))
            all_colors.append(list(base_colors[l]))

            sorted_up_to_idx += 1

        return all_l, all_colors

    @staticmethod
    def __bubble_sort_with_colors(l: list, base_colors: list) -> (list, list):

        # done_idx denotes the index we need to end consideration at. Bubble sort works such that after a while,
        # some of the largest numbers/rightmost indices will be sorted, so we don't need to consider those anymore.
        # Everything after done_idx is considered to be fully sorted

        all_l = [list(l)]
        all_colors = [list(base_colors[l])]
        length_l = len(l)

        for done_idx in range(length_l, -1, -1):

            # When this inner for loop breaks, we've finished 1 pass across the data going from left to right
            for starting_idx in range(0, done_idx-1):
                if l[starting_idx] > l[starting_idx+1]:
                    l[starting_idx:starting_idx+2] = [l[starting_idx+1], l[starting_idx]]

                    all_l.append(list(l))
                    all_colors.append(list(base_colors[l]))

        return np.array(all_l), all_colors

    def sort_and_make_gif(self, sorting_method: str, length: int=30, max_n: int=50, seed: int=10, save_fig: bool=False):
        np.random.seed(seed)
        sorting_method = sorting_method.strip().lower()

        allowed_sorting_methods = ["insertion", "bubble", "selection"]
        if sorting_method not in allowed_sorting_methods:
            raise ValueError(f"The sorting method is set to {sorting_method}. Only {allowed_sorting_methods} are allowed.")

        if length > 100:
            raise ValueError(f"Please be sensible with the number of bars in the bar plot, and set length to < 100. Currently, it's set to {length}.")
        if max_n < 1:
            raise ValueError(f"max_n needs to be >= 1. It's currently set to {max_n}.")

        a = np.random.randint(low=1, high=max_n+1, size=(length)).tolist()
        base_colors = cm.rainbow(np.linspace(0, 1, max_n + 1))

        if sorting_method == "insertion":
            all_l, all_colors = self.__insertion_sort_with_colors(a.copy(), base_colors)
        elif sorting_method == "bubble":
            all_l, all_colors = self.__bubble_sort_with_colors(a.copy(), base_colors)
        elif sorting_method == "selection":
            all_l, all_colors = self.__selection_sort_with_colors(a.copy(), base_colors)

        # TODO: Make dependencies explicit. Once AnimateBarPlot class is in a utils file, this will be much easier as we'll import the class
        AnimateBarPlot().animate(all_l, all_colors, max_n+1, f"{sorting_method.capitalize()} Sort", save_fig)

class SortingAlgorithms:
    @staticmethod
    def insertion_sort(arr: list) -> list:

        length = len(arr)
        for done_idx in range(length-1):

            starting_idx = done_idx+1
            while arr[starting_idx] < arr[starting_idx-1] and starting_idx > 0:
                arr[starting_idx-1:starting_idx+1] = [arr[starting_idx], arr[starting_idx-1]]

                starting_idx -= 1

        return arr

    @staticmethod
    def bubble_sort(arr: list) -> list:

        # done_idx denotes the index we need to end consideration at. Bubble sort works such that after a while,
        # some of the largest numbers/rightmost indices will be sorted, so we don't need to consider those anymore.
        # Everything after done_idx is considered to be fully sorted

        length = len(arr)

        for done_idx in range(length, -1, -1):

            # When this inner for loop breaks, we've finished 1 pass across the data going from left to right
            for starting_idx in range(0, done_idx-1):
                if arr[starting_idx] > arr[starting_idx+1]:
                    arr[starting_idx:starting_idx+2] = [arr[starting_idx+1], arr[starting_idx]]

        return arr

    @staticmethod
    def __merge_two_arrays_for_merge_sort(arr1: list, arr2: list) -> list:
        """
        Assumes the arrays are already sorted
        :param arr1:
        :param arr2:
        :return:
        """
        sorted_array = []

        while len(arr1) and len(arr2):
            if arr1[0] <= arr2[0]:
                sorted_array.append(arr1[0])
                arr1 = arr1[1:]
            else:
                sorted_array.append(arr2[0])
                arr2 = arr2[1:]

        # Note that at this point, either arr1 or arr2 is empty, and the leftover values are larger than everything in the sorted_array
        sorted_array = sorted_array + arr1 + arr2
        return sorted_array

    def merge_sort(self, arr: list) -> list:
        if len(arr) == 1:
            return arr
        elif len(arr) == 2:
            return [arr[1], arr[0]] if arr[0] > arr[1] else arr
        split_idx = int(len(arr) // 2)

        left_array = arr[:split_idx]
        right_array = arr[split_idx:]

        left_array2 = self.merge_sort(left_array)
        right_array2 = self.merge_sort(right_array)

        return self.__merge_two_arrays_for_merge_sort(left_array2, right_array2)

    @staticmethod
    def selection_sort(arr: list) -> list:
        sorted_up_to_idx = 0
        length = len(arr)

        while sorted_up_to_idx < length:
            min_element = arr[sorted_up_to_idx]
            min_idx = sorted_up_to_idx
            for i in range(sorted_up_to_idx, length):
                if arr[i] < min_element:
                    min_element = arr[i]
                    min_idx = i

            arr[min_idx] = arr[sorted_up_to_idx]
            arr[sorted_up_to_idx] = min_element

            sorted_up_to_idx += 1

        return arr

    def quick_sort(self, arr: list) -> list:
        # List is small enough to do this
        if len(arr) <= 1:
            return arr
        elif len(arr) == 2:
            return [arr[1], arr[0]] if arr[0] > arr[1] else arr

        # We pivot about the first number in the list and split into elements < pivot, elements = pivot, and elements > pivot.
        # We include the middle section (elements = pivot) because they don't actually need to be sorted along with the other numbers. We already know where they go.
        partition_num = arr[0]

        partition_left = []
        partition_middle = []
        partition_right = []

        for i in arr:
            if i < partition_num:
                partition_left.append(i)
            elif i == partition_num:
                partition_middle.append(i)
            else:
                partition_right.append(i)

        return self.quick_sort(partition_left) + partition_middle + self.quick_sort(partition_right)

class AnalyzeSortingAlgorithms(SortingAlgorithms):

    def compare_sorting_algorithm_times(self, max_n: int = 1e6, lengths: list[int] = [100, 500, 1000]) -> pd.DataFrame:
        """
        This function creates a dataframe showing how long each sorting algorithm took to sort a list of each length in lengths.

        :param max_n: The highest possible number in the random list to be generated and sorted
        :param lengths: The lengths of lists to generate and sort
        :return: A dataframe showing how long each sorting algorithm took to sort lists of each length in lengths
        """

        sort_algorithms = {"insertion": super.insertion_sort, "bubble": super.bubble_sort, "merge": super.merge_sort, "quick": super.quick_sort, "selection": super.selection_sort}
        times_df = pd.DataFrame(index=lengths, columns=list(sort_algorithms.keys()))

        for length in lengths:
            np.random.seed(random.randint(100, 10000))
            a = np.random.randint(low=1, high=int(max_n) + 1, size=(int(length))).tolist()

            for algo_name, function_name in sort_algorithms.index:

                start_time = time.time()
                _ = function_name(a.copy())
                end_time = time.time()
                times_df.loc[length, algo_name] = end_time - start_time

            print(f"Length {length} done")

        return times_df
