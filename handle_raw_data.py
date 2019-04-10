# import external dependencies
import pandas as pd
import numpy as np
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import internal modules
import Append_Owner_Address_Classifiers as aoac
import CategoryPercentage as cp

def one_hot_data(data):
    # one-hot data
    one_hot_columns = ['zip']
    for col in one_hot_columns:
        uniques = data[col].unique()
        for u in uniques:
            data[f"{col}_{u}"] = (data[col] == u).astype(np.int32)
    data.drop(columns=one_hot_columns, inplace=True)
    return data

def trim_data():

    print("\ntrimming data...")

    #===========================================================================
    # eco district
    #===========================================================================

    # load data
    data = pd.read_csv('data/raw_data/EDNPI19.csv', low_memory=False)

    # select useful columns
    data = data[['PARCEL', 'LAT', 'LON', 'Z', 'LANDSQFT', 'NLRNBHD']]
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

    #===========================================================================
    # rental
    #===========================================================================

    # load data
    data = pd.read_csv('data/raw_data/Rental.csv')

    # list to drop
    drop_cols = ['TaxPayer', 'YearBuilt', 'AgentName"AgentContact', 'AgentCity', 'AgentState','AgentZip', 'AgentZip4']

    # drop columns
    data.drop(columns=drop_cols, inplace=True)

    # rename
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv("data/trimmed/rental.csv", index=False)

    #===========================================================================
    # geographic
    #===========================================================================

    # load data
    data = pd.read_csv("data/raw_data/geographic.csv", index_col="Unnamed: 0")

    # only seclect desired columns
    data = data[['parcel', 'MinDistHosp/Miles']]

    # rename columns to lower case
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv("data/trimmed/geographic.csv", index=False)

def join_trimmed_data():

    print("\njoining data")

    #===========================================================================
    # set up main data
    #===========================================================================

    # get starting DataFrame to join to -- pick smallest one
    JOIN = pd.read_csv("data/clean_residential_and_sale.csv", low_memory=False, index_col="Unnamed: 0")
    # rename columns to lower case
    JOIN = JOIN.rename(str.lower, axis='columns')

    #===========================================================================
    # inner join: 'eco', 'notice', 'parcel', and 'geographic'
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
    JOIN = join_it(JOIN,'geographic')

    #===========================================================================
    # left join: 'Rental'
    #===========================================================================

    print("\nmerging Rental")

    # load rental data
    rental = pd.read_csv("data/trimmed/rental.csv")

    # left join
    JOIN = JOIN.merge(rental, how='left', on='parcel')

    #===========================================================================

    # save
    print("\nsaving as 'data/dirty.csv'")
    JOIN.to_csv("data/dirty.csv", index=False)

def clean_and_transform():

    print("\ncleaning and transforming data...")

    # load data
    data = pd.read_csv("data/dirty.csv")

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
        "personalproperty_yes",
        "deed_quit claim deed",
        "buyersellerrelated_yes",
        "partialinterest_yes",
        "financing_cash"
        ]

    # put columns into lower case
    filter_cols = [x.lower() for x in filter_cols]

    #filter out the observations where the filter cols == 1
    for col in filter_cols:
        data = data[data[col] == 0]

    # filter data where rooms == 99
    data = data[data.rooms != 99]

    # filter out observations where longitude < -112
    data = data[data.lon > -112]

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
        'validationdescription_correction of previously recorded deed                 ',
        "personalproperty_no",
        "heat_2.0",
        "deed_contract or agreement",
        "roof_5.0",
        "heat_3.0",
        "cool_9.0",
        "heat_9.0",
        "heat_4.0",
        "heat_0.0",
        "partialinterest_no",
        "buyersellerrelated_no",
        "financing_other",
        "landfcv",
        "age"
        ]

    # put column names into lower case
    drop_cols = [x.lower() for x in drop_cols]

    # drop the columns
    data.drop(columns=drop_cols+filter_cols, inplace=True)
    # data = data.drop(columns=drop_cols+filter_cols)

    #===========================================================================
    # cleaning
    #===========================================================================

    #===========================================================================
    # transform
    #===========================================================================

    aoac.fix_rental(data)

    #===========================================================================
    # save
    #===========================================================================

    data.to_csv("data_sets/clean.csv", index=False)

def split_into_sets(column):

    print("\nsplitting data into 'train', 'validate', and 'test' sets...")

    # load cleaned data
    data = pd.read_csv("data_sets/clean.csv")

    # drop parcel
    data.drop(columns=['parcel'], inplace=True)

    # get an array of the unique values in the DataFrame column
    uniques = data[column].unique()

    # loop through each unique value in the column
    for val in uniques:
        data[f"{column}_{val}"] = (data[column] == val).to_numpy(dtype=np.int32)

    # delete the original column
    data.drop(column, axis=1, inplace=True)

    # shuffle the entire set of observations
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)

    # create impty DataFrames for the data sets
    train = pd.DataFrame()
    validate = pd.DataFrame()
    test = pd.DataFrame()

    # stratify the data
    for val in uniques:

        # get the data subset
        df = data[data[f"{column}_{val}"] == 1]
        # find the length of the subset
        N = df.shape[0]

        # get the number of observations to go into each data set
        N_train = int(.6*N)
        N_validate = int(.2*N)
        N_test = N - N_validate - N_train

        # append the sub data sets to the data sets
        train = train.append(df[ :N_train ], ignore_index=True)
        validate = validate.append(df[N_train : N_train+N_validate ], ignore_index=True)
        test = test.append(df[ N_train+N_validate: ], ignore_index=True)

    # save the dataframes
    train.to_csv("data_sets/train.csv", index=False)
    validate.to_csv("data_sets/validate.csv", index=False)
    test.to_csv("data_sets/test.csv", index=False)

def run():

    # trim_data()
    # join_trimmed_data()
    clean_and_transform()
    split_into_sets('nlrnbhd')
    # cp.check_categories("data_sets/clean.csv")

#===============================================================================
# figures
#===============================================================================

def plot_lon_lat():
    dirty = pd.read_csv("data/dirty.csv")
    clean = pd.read_csv("data_sets/clean.csv")

    # is in both data
    both = dirty[dirty.parcel.isin(clean.parcel)]
    # was dropped
    dropped = dirty[~dirty.parcel.isin(clean.parcel)]

    fig,ax = plt.subplots()
    fig.suptitle("Looking at Houses that were Dropped")
    ax.scatter(both.lon.values, both.lat.values, c='g', alpha=.1, label='both')
    ax.scatter(dropped.lon.values, dropped.lat.values, c='r', alpha=.1, label='dropped')
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1)
    plt.legend()

    fig.savefig("lon_lat.pdf")
    plt.close(fig)

    pass
