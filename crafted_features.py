def calc_features(ds):
    features_df_struct = [{"name": "n_words", "func": count_words},
                          {"name": "n_unique_words", "func": count_unique_words},
                          {"name": "n_capital_letters", "func": count_capitalized_letters},
                          ]
    ds = load_data(filename, column_names)

    for i in range(len(features_df_struct)):
        column_structure = features_df_struct[i]
        ds[column_structure["name"]] = column_structure["func"](ds["text"])
        ds.reset_index(drop=True, inplace=True)
    return ds


def count_words(text):
    return text.apply(lambda x: len(x))


def count_unique_words(text):
    return text.apply(lambda x: len(set(x)))
