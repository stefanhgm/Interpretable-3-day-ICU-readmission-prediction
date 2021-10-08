import argparse

from helper.io import read_item_processing_descriptions_from_excel


def main():
    parser = argparse.ArgumentParser(description='Script to get an overview of variables and pharmaceuticals.')
    parser.add_argument('item_overview', type=str, help='Description of all PDMS items and generated variables.')
    args, _ = parser.parse_known_args()
    item_processing = read_item_processing_descriptions_from_excel(args.item_overview)
    item_processing['processed'] = False

    # Parse included variables
    print("Included variables")
    for idx, _ in item_processing.loc[(item_processing['decision'] == 'included') &
                                      (item_processing['QS_ITEMTYPE'] != 10)].iterrows():
        variable = item_processing.loc[idx]
        item_processing.loc[idx, 'processed'] = True
        features = variable['feature_generation'].replace('[', '').replace(']', '').replace('"', '')
        print(f"{variable['variable_name']}; {variable['type']}; "
              f"{variable['id']};{variable['QS_NAME']};{variable['total usages']};{features}")
    print("")

    # Parse merged variables
    print("Merged variables")
    for idx, _ in item_processing.loc[(item_processing['id'] >= 10000) & (item_processing['id'] < 20000)].iterrows():
        merged_variable = item_processing.loc[idx]
        item_processing.loc[idx, 'processed'] = True
        merged_name = merged_variable['variable_name']
        features = merged_variable['feature_generation'].replace('[', '').replace(']', '').replace('"', '')
        print(f"{merged_variable['variable_name']} ({merged_variable['type']}, {features})")

        for jdx, _ in item_processing.loc[(item_processing['decision'] == 'merged') &
                                          (item_processing['variable_name'] == merged_name)].iterrows():
            variable = item_processing.loc[jdx]
            item_processing.loc[jdx, 'processed'] = True
            print(f"{variable['id']};{variable['QS_NAME']};{variable['total usages']}")
    print("")

    print("Excluded non-pharmaceutical variables")
    for idx, _ in item_processing.loc[(~item_processing['processed']) & (item_processing['id'] < 10000) &
                                      (item_processing['QS_ITEMTYPE'] != 10)].iterrows():
        variable = item_processing.loc[idx]
        print(f"{variable['id']}; {variable['QS_NAME']}; {variable['total usages']}")

    # Parse non-excluded pharmaceuticals.
    print("Pharmaceutical variables")
    pharmaceuticals = item_processing.loc[item_processing['QS_ITEMTYPE'] == 10]
    old_num_pharmaceuticals = pharmaceuticals.shape[0]
    pharmaceuticals = pharmaceuticals.loc[~(pharmaceuticals['decision'].str.contains('excluded', na=False))]
    pharmaceuticals['in_cluster'] = False
    clusters = item_processing.loc[(item_processing['id'] >= 20000)]
    clusters.sort_values(['variable_name'], ascending=[True], inplace=True, ignore_index=True)
    pharmaceuticals.sort_values(['ATC'], ascending=[True], inplace=True, ignore_index=True)

    for idx, _ in clusters.iterrows():
        variable_description = clusters.loc[idx]
        item_id, var_name, var_type = variable_description.loc[['id', 'variable_name', 'type']]
        atc_prefix = var_name.split()[0]
        print(f"{var_name}, {atc_prefix}")

        for jdx, _ in pharmaceuticals.loc[pharmaceuticals['ATC'].str.contains('^(?:' + atc_prefix + ')', na=False)]\
                .iterrows():
            variable = pharmaceuticals.loc[jdx]
            if variable['in_cluster'] and not var_name.endswith(' Antihypertensiva'):
                print(f"Variable used twice: {variable['id']}, {variable['ATC']}")
                raise Exception("Medication used twice")
            if not var_name.endswith(' Antihypertensiva'):  # Exclude super category from twice usage check.
                pharmaceuticals.loc[jdx, 'in_cluster'] = True
            print(f"\t{variable['id']};{variable['QS_NAME']};{variable['total usages']};"
                  f"{variable['ATC']};{variable['ATC-name']};{variable['ATC-path']}")
    print("")

    print("Excluded pharmaceutical variables")
    for idx, _ in pharmaceuticals.loc[~pharmaceuticals['in_cluster']].iterrows():
        variable = pharmaceuticals.loc[idx]
        print(f"{variable['id']};{variable['QS_NAME']};{variable['total usages']};"
              f"{variable['ATC']};{variable['ATC-name']};{variable['ATC-path']}")

    print("")
    print(f"Excluded pharmaceuticals: {old_num_pharmaceuticals - pharmaceuticals.shape[0]}, "
          f"in cluster: {pharmaceuticals.loc[pharmaceuticals['in_cluster']].shape[0]}, "
          f"not in cluster: {pharmaceuticals.loc[~pharmaceuticals['in_cluster']].shape[0]}")

    # Determine decision for pharmaceuticals based on their belonging to a cluster.
    # for idx, variable in item_processing.loc[(item_processing['QS_ITEMTYPE'] == 10) |
    #                                          (item_processing['QS_ITEMTYPE'] == 11)].iterrows():
    #     print(str(variable['id']) + ';', end='')
    #     if variable['id'] not in pharmaceuticals['id'].values:
    #         print(variable['decision'])
    #     elif variable['id'] in pharmaceuticals.loc[pharmaceuticals['in_cluster'], 'id'].values:
    #         print('merged')
    #     else:
    #         print('excluded irrelevant', "NOT IN CLUSTER")


if __name__ == '__main__':
    main()
