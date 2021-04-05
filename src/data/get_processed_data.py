import numpy as np
import pandas as pd

def read_data():
    # set the path of the raw data
    raw_data_path = os.path.join(os.path.pardir, 'src', 'data', 'raw')
    # print(raw_data_path)
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    # read the path will all default parameters
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col='PassengerId')
    test_df['Survived'] = -888
    df = pd.concat((trains_df, test_df), axis=0)
    return df


def process_data(df):
    #using the method chaining concept
    return (df
            # create title feature - then add this
            .assign(Title = lambda x : x.Name.map(get_title))
            # working with missing values - start with this
            .pipe(fill_missing_values)
            # create fare bin feature
            .assign(Fare_Bin = lambda x : pd.qcut(x.Fare, 4 , labels=['very_low', 'low', 'high', 'very_high']))
            # create features age_state, family_size, is_mother & deck
            .assign(Age_State = lambda x : np.where(x.Age >= 18, 'Adult', 'Child'))
            .assign(Family_Size = lambda x : x.Parch + x.SibSp + 1)
            .assign(Is_Mother = lambda x : np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age > 18) & (df.Title != 'Miss')), 1, 0)
            .assign(Cabin = lambda x : np.where(x.Cabin == 'T', np.nan, x.Cabin))
            .assign(Deck = lambda x : x.Cabin.map(get_deck))
            # feature encoding
            .assign(IsMale = np.where(df.Sex == 'male', 1, 0))
            .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'FareBin', 'Embarked', 'AgeState'])
            # add code to drop unnecessary columns
            .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
            # reorder columns
            .pipe(reorder_columns)
           )

def get_title(name):
    title_group = {
        'mr': 'Mr',
        'mrs': 'Mrs',
        'miss': 'Miss',
        'master': 'Master',
        'don': 'Sir',
        'rev': 'Sir',
        'dr': 'Officer',
        'mme': 'Mrs',
        'ms': 'Mrs',
        'major': 'Officer',
        'lady': 'Lady',
        'sir': 'Sir',
        'mlle': 'Miss',
        'col': 'Officer',
        'capt': 'Officer',
        'the countess': 'Lady',
        'jonkheer': 'Sir',
        'dona': 'Lady'
    }
    first_name_title = name.split(',')[1]
    title = first_name_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def fill_missing_values(df):  
    # replacing missing values
    # embarked
    df.embarked.fillna('C', inplace=True)
    # fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    # age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)
    return df

def reorder_columns(df):
    columns = [column for column in new_df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    new_df = new_df[columns]
    return df

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')      

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, 'src', 'data', 'processed')
    write_train_path = os.path.join(processed_data_path, 'train.csv')
    write_test_path = os.path.join(processed_data_path, 'test.csv')
    # train_data
    df[df.Survived != -888].to_csv(write_train_path)
    # test
    columns = [column for column in new_df.columns if column != 'Survived']        
    df[df.Survived == -888][columns].to_csv(write_test_path)     

def __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
