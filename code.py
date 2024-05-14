# Importing required modules
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import math
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

# Base setup for SQLAlchemy
Base = declarative_base()

# Classes for Database exception handling
class DatabaseError(Exception):
    """Custom Exception for Database errors"""
    pass

class DataLoaderError(Exception):
    """Custom Exception for DataLoader errors"""
    pass

# Data Loader Class
class DataLoader:
    """Load and organize data from CSV files."""
    
    # Constructor
    def __init__(self, train_path, ideal_path, test_path):
        """
        Initialize the DataLoader with paths to training, ideal, and testing data.
        """
        self.train_path = train_path
        self.ideal_path = ideal_path
        self.test_path = test_path
        self.train_data = None
        self.ideal_data = None
        self.test_data = None

    # Function to read CSV files
    def load_data(self):
        """
        Load training, ideal, and testing datasets.
        """
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.ideal_data = pd.read_csv(self.ideal_path)
            self.test_data = pd.read_csv(self.test_path)
        except Exception as e:
            raise DataLoaderError(f"Error loading data: {e}")

# Classes to define the Structures of tables in database
# It is also inherting Base class
class TrainingData(Base):
    # Adding columns and creating structure
    """SQLAlchemy class for training data table."""
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)

class IdealFunctions(Base):
    """SQLAlchemy class for ideal functions table."""
    __tablename__ = 'ideal_functions'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y_values = {f"y{i+1}": Column(Float) for i in range(50)}

# Incorporate columns dynamically
for attr, column in IdealFunctions.y_values.items():
    setattr(IdealFunctions, attr, column)

class TestData(Base):
    """SQLAlchemy class for test data table."""
    __tablename__ = 'test_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    delta_y = Column(Float)
    ideal_func_no = Column(Integer)

# Class to add the data to SQLite Database
class DatabaseManager:
    """Manage database interactions using SQLAlchemy."""
    def __init__(self, db_url='sqlite:///data.db'):
        """
        Initialize the database manager and set up an SQLAlchemy session.
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    # Function to add data
    def add_data(self, data, table_class):
        """
        Add data to a database table.

        Args:
            data (pd.DataFrame): Data to be added.
            table_class (SQLAlchemy Model): Target SQLAlchemy class representing the table.
        """
        try:
            # Iterating Rows
            for index, row in data.iterrows():
                entry = table_class(**row.to_dict())
                
                # Adding row to database
                self.session.add(entry)
            self.session.commit()
        except Exception as e:
            raise DatabaseError(f"Error adding data to database: {e}")

# Class to perform the second task
# that is to select the ideal function
class IdealFunctionSelector:
    """Class to select the best-fitting ideal functions for training data."""
    def __init__(self, train_data, ideal_data, num_functions=4):
        """
        Initialize the selector with training and ideal data.

        Args:
            train_data (pd.DataFrame): Training dataset.
            ideal_data (pd.DataFrame): Ideal dataset.
            num_functions (int): Number of ideal functions to select.
        """
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.num_functions = num_functions
        self.results = {}

    # SSD Calculatin function
    def calculate_ssd(self, train_col, ideal_col):
        """
        Calculate the sum of squared deviations.

        Args:
            train_col (str): Training column name.
            ideal_col (str): Ideal column name.

        Returns:
            float: Sum of squared deviations.
        """
        ssd = np.sum((self.train_data[train_col] - self.ideal_data[ideal_col]) ** 2)
        return ssd

    def select_best_ideal_functions(self):
        """
        Select the best ideal functions based on sum of squared deviations.

        Returns:
            list: List of best ideal function names.
        """
        for column in self.ideal_data.columns[1:]:
            total_ssd = 0
            for train_col in ['y1', 'y2', 'y3', 'y4']:
                ssd = self.calculate_ssd(train_col, column)
                total_ssd += ssd
            self.results[column] = total_ssd

        best_functions = sorted(self.results, key=self.results.get)[:self.num_functions]
        return best_functions

# Test matcher class to
# match the selected fucntions with test data
class TestMatcher:
    """Class to match test data points to ideal functions."""
    def __init__(self, test_data, train_data, ideal_data, selected_functions):
        """
        Initialize with test data, training data, ideal data, and selected functions.

        Args:
            test_data (pd.DataFrame): Test dataset.
            train_data (pd.DataFrame): Training dataset.
            ideal_data (pd.DataFrame): Ideal dataset.
            selected_functions (list): List of selected ideal functions.
        """
        self.test_data = test_data
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.selected_functions = selected_functions
        self.results = []

    def calculate_max_deviations(self):
        """
        Calculate the maximum deviations for the training data.

        Returns:
            dict: Dictionary with function names and their maximum allowed deviations.
        """
        max_deviations = {}
        for train_col, func in zip(['y1', 'y2', 'y3', 'y4'], self.selected_functions):
            max_deviation = np.max(np.abs(self.train_data[train_col] - self.ideal_data[func]))
            max_deviations[func] = max_deviation * math.sqrt(2)
        return max_deviations

    def match_test_data(self):
        """
        Match test data points to the selected ideal functions.

        Returns:
            pd.DataFrame: Matched results including deviations.
        """
        max_deviations = self.calculate_max_deviations()

        for index, row in self.test_data.iterrows():
            x, y = row['x'], row['y']
            min_deviation = float('inf')
            assigned_func = None

            for func in self.selected_functions:
                ideal_y = np.interp(x, self.ideal_data['x'], self.ideal_data[func])
                deviation = abs(y - ideal_y)

                if deviation <= max_deviations[func] and deviation < min_deviation:
                    min_deviation = deviation
                    assigned_func = func

            self.results.append({
            'x': x,  
            'y': y, 
            'delta_y': min_deviation if assigned_func else None,
            'ideal_func_no': assigned_func  
        })


        return pd.DataFrame(self.results)

# Data Visualization class
class DataVisualizer:
    """Class for data visualization using Bokeh."""
    @staticmethod
    def plot_functions(training, ideal, selected_functions):
        """
        Plot training and selected ideal functions.

        Args:
            training (pd.DataFrame): Training dataset.
            ideal (pd.DataFrame): Ideal dataset.
            selected_functions (list): List of selected ideal function names.

        Returns:
            bokeh.plotting.figure: Bokeh plot object.
        """
        p = figure(title="Training Data and Selected Ideal Functions", x_axis_label='x', y_axis_label='y', width=800, height=400)
        colors = ['red', 'green', 'blue', 'purple']
        for i, col in enumerate(['y1', 'y2', 'y3', 'y4']):
            p.circle(training['x'], training[col], legend_label=f'Training {col}', color=colors[i], size=5)
        for i, func in enumerate(selected_functions):
            p.line(ideal['x'], ideal[func], legend_label=f'Ideal {func}', color=colors[i], line_width=2)
        p.legend.click_policy = "hide"
        return p

    @staticmethod
    def plot_test_data(test_data, ideal):
        """
        Plot test data points and their deviations.

        Args:
            test_data (pd.DataFrame): Test dataset with deviations.
            ideal (pd.DataFrame): Ideal dataset.

        Returns:
            bokeh.plotting.figure: Bokeh plot object.
        """
        p = figure(title="Test Data and Deviations", x_axis_label='x', y_axis_label='y', width=800, height=400)
        p.circle(test_data['x'], test_data['y'], legend_label='Test Data', color='gray', size=10)
        for index, row in test_data.iterrows():
            if pd.notna(row['ideal_func_no']):
                ideal_y = np.interp(row['x'], ideal['x'], ideal[row['ideal_func_no']])
                p.line([row['x'], row['x']], [row['y'], ideal_y], legend_label=row['ideal_func_no'], color='orange', line_width=2)
        p.legend.click_policy = "hide"
        return p


# Applying all fucntion and classes on data
if __name__ == "__main__":
    
    # Loading data files
    data_loader = DataLoader('Datasets1/train.csv', 'Datasets1/ideal.csv', 'Datasets1/test.csv')
    data_loader.load_data()
    train_data, ideal_data, test_data = data_loader.train_data, data_loader.ideal_data, data_loader.test_data
    
    # Adding data to DB
    db_manager = DatabaseManager()
    db_manager.add_data(train_data, TrainingData)
    
    # Selecting Ideal Functions
    ideal_selector = IdealFunctionSelector(train_data, ideal_data)
    selected_funcs = ideal_selector.select_best_ideal_functions()
    
    # Performing testing
    test_matcher = TestMatcher(test_data, train_data, ideal_data, selected_funcs)
    matched_data = test_matcher.match_test_data()
    db_manager.add_data(matched_data, TestData)
    
    # Making visualizations
    plot1 = DataVisualizer.plot_functions(train_data, ideal_data, selected_funcs)
    plot2 = DataVisualizer.plot_test_data(matched_data, ideal_data)
    show(gridplot([[plot1], [plot2]]))