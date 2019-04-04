import pandas as pd
import numpy as np

def one_hot_data(data):
    # one-hot data
    one_hot_columns = ['zip']
    for col in one_hot_columns:
        uniques = data[col].unique()
        for u in uniques:
            data[f"{col}_{u}"] = (data[col] == u).astype(np.int32)
    data.drop(columns=one_hot_columns, inplace=True)
    return data

def check_categories(filename):

    # load data
    df = pd.read_csv(filename)

    # Number of observations.
    N,D = df.shape

    fractions = []

    for col in df.columns:
        # Get categorical columns.
        if set(df[col]) <= {0, 1}:
            # (column name, amount true)
            S = df[col].sum()
            fractions.append( (col, S/N, S) )

    fractions.sort(key=lambda x: x[1])

    print("\nCategorical Data\nPercentage : N_observations : Category")
    for i in fractions:
        print("{:0.4f} : {} : {}".format(i[1]*100,i[2],i[0]))

def trim_data():

    #===========================================================================
    # eco district
    #===========================================================================

    # load data
    data = pd.read_csv('data/raw_data/EDNPI19.csv', low_memory=False)

    # select useful columns
    data = data[['PARCEL', 'LAT', 'LON', 'Z', 'LANDSQFT']]
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv('data/trimmed/eco.csv', index=False)

    #===========================================================================
    # notice of value
    #===========================================================================

    # load data
    data = pd.read_csv(f"data/raw_data/Notice19.csv", sep=',', encoding='latin', dtype='str', low_memory=False)
    for col in data: data[col].astype(int, errors='ignore')

    # select useful columns
    data = data[['PARCEL', 'LANDFCV']] #ZIP
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv('data/trimmed/notice.csv', index=False)

    #===========================================================================
    # parcel model data -- just get latest set
    #===========================================================================

    # load data
    data = pd.read_csv('data/raw_data/PclModelData19.csv', low_memory=False)

    # select useful columns
    data = data[['Parcel', 'Age']]
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv('data/trimmed/parcel.csv', index=False)

def join_trimmed_data():

    #===========================================================================
    # set up main data
    #===========================================================================

    # get starting DataFrame to join to -- pick smallest one
    JOIN = pd.read_csv("data/clean_residential_and_sale.csv", low_memory=False, index_col="Unnamed: 0")
    # rename columns to lower case
    JOIN = JOIN.rename(str.lower, axis='columns')

    #===========================================================================
    # inner join: 'eco', 'notice', and 'parcel'
    #===========================================================================

    # make a function to join to JOIN with filename as input
    def join_it(JOIN,file):

        print(f"\nmerging {file}")

        # get the full file name
        filename = f'data/trimmed/{file}.csv'

        # load the dataframe
        df = pd.read_csv(filename, low_memory=False)

        # perform inner join
        JOIN = JOIN.merge(df, how='inner', on='parcel')

        # delete df
        del df

        # output
        return JOIN

    # go through and join_it for each of the 'eco', 'notice' and 'parcel' data
    JOIN = join_it(JOIN,'eco')
    JOIN = join_it(JOIN,'notice')
    JOIN = join_it(JOIN,'parcel')

    #===========================================================================
    # left join: 'Rental'
    #===========================================================================

    print("\nmerging Rental")

    # load rental data
    rental = pd.read_csv("data/trimmed/Rental.csv")

    # rename columns
    rental = rental.rename(str.lower, axis='columns')

    # left join
    JOIN = JOIN.merge(rental, how='left', on='parcel')

    #===========================================================================

    # save
    print("\nsaving as 'data/dirty.csv'")
    JOIN.to_csv("data/dirty.csv", index=False)

def clean_and_transform():

    # load data
    data = pd.read_csv("data/dirty.csv", low_memory=False)

    #===========================================================================
    # filter
    #===========================================================================

    # list of columns to filter out: they contain sales that do not accurately reflect fair market value
    filter_cols = [
        "ValidationDescription_Buyer/Seller are related parties or corporate entities",
        "ValidationDescription_Buyer/Seller is a Non-Profit institution",
        "ValidationDescription_More than five (5) parcels being sold",
        "ValidationDescription_Sale includes quantifiable Personal Property > 5%",
        "ValidationDescription_Sale includes unquantifiable Personal Property > 5%",
        "ValidationDescription_Sale involves exchange or trade",
        "ValidationDescription_Sale price missing",
        "ValidationDescription_Sale pursuant to a court order",
        "ValidationDescription_Sale to or from a government agency",
        "ValidationDescription_Sale under duress",
        "ValidationDescription_Suspicious Sale",
        "ValidationDescription_Unsecured Mobile Home",
        "ValidationDescription_Unusable sale which does not fit any other reject codes",
        "ValidationDescription_Trust sale of nominal consideration or convenience",
        "ValidationDescription_Sale of convenience for nominal consideration",
        "personalproperty_yes"
        ]

    # put columns into lower case
    filter_cols = [x.lower() for x in filter_cols]

    #filter out the observations where the filter cols == 1
    for col in filter_cols:
        data = data[data[col] == 0]

    #===========================================================================
    # drop
    #===========================================================================

    # list of columns to drop
    drop_cols = [
        "ValidationDescription_Buyer/Seller has an Out-Of-State Address",
        # "ValidationDescription_Correction of previously recorded deed",
        "ValidationDescription_Good Sale",
        "ValidationDescription_Internet sale",
        "ValidationDescription_Improvements not yet on assessment roll",
        "ValidationDescription_Name/Address of Buyer or Seller is missing",
        "ValidationDescription_Payoff of Land contract",
        "ValidationDescription_Property altered since date of sale",
        "ValidationDescription_Property type/use code are not consistent",
        "ValidationDescription_Sale of partial interest",
        "ValidationDescription_Split legislative class/assessment ratio",
        "validationdescription_correction of previously recorded deed",
        "personalproperty_no"
        ]

    # put column names into lower case
    drop_cols = [x.lower() for x in drop_cols]

    # drop the columns
    data.drop(columns=drop_cols+filter_cols, inplace=True)

    #===========================================================================
    # cleaning
    #===========================================================================

    #===========================================================================
    # save
    #===========================================================================

    data.to_csv("data/clean.csv", index=False)

def split_into_sets():

    print("\nsplitting data into 'train', 'validate', and 'test' sets...")

    # load cleaned data
    data = pd.read_csv("data/clean.csv")

    # shuffle the entire set of observations
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)

    # get the numbers of observations for the data sets
    N = data.shape[0]
    N_train = int(.6*N)
    N_validate = int(.2*N)
    N_test = N - N_validate - N_train

    # get the cross validation data
    train = data[ :N_train ]
    validate = data[N_train : N_train+N_validate ]
    test = data[ N_train+N_validate: ]

    # save the dataframes
    train.to_csv("data_sets/train.csv", index=False)
    validate.to_csv("data_sets/validate.csv", index=False)
    test.to_csv("data_sets/test.csv", index=False)

def run():

    # trim_data()
    # join_trimmed_data()
    clean_and_transform()
    split_into_sets()
