import numpy as np
import pandas as pd
import sklearn as sk

def convert_gender(gender):
    if gender == 'M':
        return 0
    else: # female 
        return 1 

def convert_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    else:
        return 5 
    
def convert_city_category(city):
    if city == 'A':
        return 0
    elif city == 'B':
        return 1
    else:
        return 2
    
def convert_stay_in_city_years(stay_in_years):
    if stay_in_years == '4+':
        return 4
    else:
        return int(stay_in_years)

def load_black_friday(trainsize=3000, testsize=10000):
    dataset = pd.read_csv('BlackFriday.csv')

    #print(dataset.info())
    
    ### Dealing with the NaN values ###
    # They only appear under Product Category. 
    # A plausible explanation would be that these purchases did not include said categories, so
    # we'll say there's 0 items of said category that were bought 
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.fillna.html
    dataset.fillna(0, inplace=True)
    
    #print(dataset.info)
    
    # User id and product id (?) are keys and not features, so they won't be of much use to us.
    # https://stackoverflow.com/questions/25695878/dataframe-drop-duplicates-and-dataframe-drop-not-removing-rows
    dataset.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)
    #print(dataset.info())
    
    
    # We'll convert the object types in the data to be integer types for ease of use. 
    # So we'll change gender, age, city, and stay_in_current_city_years as they all are type object
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html
    dataset['Gender'] = dataset['Gender'].apply(convert_gender)
    dataset['Age'] = dataset['Age'].apply(convert_age)
    dataset['City_Category'] = dataset['City_Category'].apply(convert_city_category)
    dataset['Stay_In_Current_City_Years'] = dataset['Stay_In_Current_City_Years'].apply(convert_stay_in_city_years)
    
    # Randomizing data to reduce bias 
    #seed = np.random.randint(0, 100)
    #dataset = sk.utils.shuffle(dataset, random_state=seed)    
    
    # Returns copy of the Purchase column
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.filter.html
    vals = dataset.filter(items=['Purchase']) 
    
    # Creating sets like in dataloader.py from assignments 
    Xtrain = dataset.iloc[0:trainsize]
    ytrain = vals.iloc[0:trainsize]
    Xtest = dataset.iloc[trainsize:trainsize+testsize]
    ytest = vals.iloc[trainsize:trainsize+testsize]
    
    Xtrain = Xtrain.drop('Purchase', axis=1)
    Xtest = Xtest.drop('Purchase', axis=1)
    
    return ((Xtrain.values,ytrain.values.ravel()), (Xtest.values,ytest.values.ravel())) 


load_black_friday()