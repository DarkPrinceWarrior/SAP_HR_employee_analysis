import numpy as np


def get_prepared_data(dataset):
    X = dataset

    # BusinessTravel
    X = X.replace('Travel_Rarely', 0)
    X = X.replace('Travel_Frequently', 1)
    X = X.replace('Non-Travel', 2)
    # Department
    X = X.replace('Research & Development', 0)
    X = X.replace('Sales', 1)
    X = X.replace('Human Resources', 2)

    # EducationField
    X = X.replace('Medical', 0)
    X = X.replace('Life Sciences', 1)
    X = X.replace('Other', 2)
    X = X.replace('Marketing', 3)
    X = X.replace('Technical Degree', 4)
    X = X.replace('Human Resources', 5)

    # Gender
    X = X.replace('Male', 0)
    X = X.replace('Female', 1)

    # JobRole
    X = X.replace('Laboratory Technician', 0)
    X = X.replace('Sales Executive', 1)
    X = X.replace('Manufacturing Director', 2)
    X = X.replace('Healthcare Representative', 3)
    X = X.replace('Sales Representative', 4)
    X = X.replace('Research Scientist', 5)
    X = X.replace('Human Resources', 6)
    X = X.replace('Manager', 7)
    X = X.replace('Research Director', 8)

    # MaritalStatus
    X = X.replace('Married', 0)
    X = X.replace('Single', 1)
    X = X.replace('Divorced', 2)

    # OverTime
    X = X.replace('No', 0)
    X = X.replace('Yes', 1)

    return X
