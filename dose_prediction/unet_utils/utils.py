#==============================================================================#
#  Author:       Michael Lempart                                                #
#  Copyright:    2021, Department of Radiation Physics, Lund University                                       #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#

#Import libraries
import csv

def get_data(csv_file):

    """

    Reads a csv file containing the training and validation set data.

    Args:
        csv_file (str): path to the csv file.
    
    Returns:
        X_train (list): training file list.
        y_train (list) training label file list.
        X_val (list): validation file list.
        y_val (list) validation label file list.

    """

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    with open(csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            line_count = 0
            for row in csv_reader:
                    if row[1] == 'training':
                            X_train.append(row[0])
                            y_train.append(row[0][-12:])
                    elif row[1] == 'validation':
                            X_val.append(row[0])
                            y_val.append(row[0][-12:])
    
    return X_train, y_train, X_val, y_val

