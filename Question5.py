import numpy as np
import csv
import matplotlib.pyplot as plt

# data column const
country = 0
date = 1
newCases = 2
population = 3
density = 4

def make_matrix(data_name, data_per_country_size, countries_size):
    countries_data = np.zeros((data_per_country_size, countries_size))
    with open(data_name) as csv_file:
        csv_reader1 = csv.reader(csv_file, delimiter=',')
        day = 0
        country_id = 0
        for row in csv_reader1:
            countries_data[day][country_id] = row[newCases]
            day += 1
            if day == 238:
                day = 0
                country_id += 1
    return countries_data

def split_first(matrix, from_x, from_y, to_x, to_y):
    return matrix[from_x:to_x, from_y:to_y]

def split_second(matrix, from_x, from_y, to_x, to_y):
    return matrix[from_x:to_x, from_y:to_y]

def least_square(A, B):
    A_transpose = np.transpose(A)
    x = np.linalg.solve(A_transpose @ A, A_transpose @ B)
    return x

def minimize_matrix(matrix, ignore_array, row_or_column_index):
    for elem in ignore_array:
        matrix = np.delete(matrix, elem, row_or_column_index)
    return matrix

def minimize_by_average_rows(matrix, average_by):
    column_size = len(matrix[0])
    row_size = len(matrix)
    curr_column = 0
    curr_row = 0
    curr_average_value = 0
    curr_average_index = 0

    average_matrix = np.zeros((int(row_size / average_by), column_size))

    while curr_column != column_size:
        curr_average_value += matrix[curr_row][curr_column]
        curr_row += 1

        if curr_row % average_by == 0:
            average_matrix[curr_average_index][curr_column] = int(curr_average_value / average_by)
            curr_average_index += 1
            curr_average_value = 0

        if curr_row == row_size:
            curr_row = 0
            curr_average_value = 0
            curr_average_index = 0
            curr_column += 1

    return average_matrix

def main():
    # Organize Country Data
    countries_data = make_matrix('CountriesData.csv', 238, 7)
    base_countries_data = split_first(countries_data, 0, 0, 189, 7)
    validation_countries_data = split_second(countries_data, 189, 0, 239, 7)
    # Organize France Data
    france_data = make_matrix('France.csv', 238, 1)
    base_France_data = split_first(france_data, 0, 0, 189, 1)
    validation_France_data = split_second(france_data, 189, 0, 239, 1)



    # Finding LS solution
    least_square_solution = least_square(base_countries_data, base_France_data)
    print("\nthis is the simple minimal LS solution: \n", least_square_solution)

    # we noticed some countries are not correlated to France, therefore we will remove them:
    ignore_countries = [6, 4, 2]
    # discarding unnecessary countries from LS solution
    least_square_solution = minimize_matrix(least_square_solution, ignore_countries, 0)

    # discarding unnecessary countries from base Countries days
    base_countries_data = minimize_matrix(base_countries_data, ignore_countries, 1)

    # discarding unnecessary countries from validation Countries days
    validation_countries_data = minimize_matrix(validation_countries_data, ignore_countries, 1)

    # evaluate Future data
    validation_France_excepted = validation_countries_data @ least_square_solution
    plt.plot(validation_France_excepted)
    plt.xlabel('days')
    plt.ylabel('New Cases')
    plt.show()

    # we noticed there are a lot of noise in the graph, we will minimize it by taking the average of every week
    base_countries_data = minimize_by_average_rows(base_countries_data, 7)
    validation_countries_data = minimize_by_average_rows(validation_countries_data, 7)
    base_France_data = minimize_by_average_rows(base_France_data, 7)
    validation_France_data = minimize_by_average_rows(validation_France_data, 7)

    # Finding LS solution again
    least_square_solution = least_square(base_countries_data, base_France_data)
    print("\nthis is the minimal LS solution after average: \n", least_square_solution)

    # evaluate Future data
    validation_France_excepted = validation_countries_data @ least_square_solution
    plt.plot(validation_France_excepted)
    plt.xlabel('days')
    plt.ylabel('New Cases')
    plt.show()

#     TODO: measure the error

if __name__ == '__main__':
    main()
