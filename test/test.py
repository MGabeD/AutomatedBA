import pandas as pd
import os
from src.core.driver import layoutProcess
from src.util import SnowflakeManager
from pandas.testing import assert_frame_equal

def assertFrameEqual(df1, df2, **kwds ):
    
    try:
        assert_frame_equal(df1,df2, check_like=True)
        return True
    except:
        return False

path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(path, 'test_data_working.csv'))

test_count = 0
success_count = 0
completion_count = 0
snowflake = SnowflakeManager()

for first_param in [2]:  # Loop over first parameter
    for second_param in [1]:  # Loop over second parameter
        results = []
        for index, row in df.iterrows():
            gt_success = False
            gpt_success = False
            database = row['db_name']
            question = row['question']
            gt_query = row['query']
            try:
                snowflake.connect(database.upper())
            except:
                continue

            test_count += 1
            try:
                gt_result = snowflake.query_df(gt_query)
                if isinstance(gt_result, pd.DataFrame):
                    gt_success = True
            except:
                pass

            try:
                # Varying the first two parameters while keeping others constant
                response, model_result = layoutProcess(str(first_param), str(second_param), '1', '1', '1', question, database.upper())
                if isinstance(model_result, pd.DataFrame):
                    if 'Snowflake Raw Error' not in model_result:

                        gpt_success = True
            except:
                pass

            if gt_success and gpt_success:
                completion_count +=1
                gt_result = gt_result.reindex(sorted(gt_result.columns), axis=1)
                model_result = model_result.reindex(sorted(model_result.columns), axis=1)
                gt_result = gt_result.sort_values(by=gt_result.columns.tolist()).reset_index(drop=True)
                model_result = model_result.sort_values(by=model_result.columns.tolist()).reset_index(drop=True)
                if assertFrameEqual(gt_result, model_result):

                    results.append((True,database, row['query_category']))
                    success_count +=1
                    print("success")
                else:
                    results.append((False, database, row['query_category']))
            else:
                results.append((False, database, row['query_category']))

        results.append(("Completion; Accuracy", str(completion_count / test_count), str(success_count / test_count)))
        result_df = pd.DataFrame(results, columns=['equality', 'database', 'query_category'])
        
        # Construct filename based on parameters
        filename = f'results_param_{first_param}_{second_param}.csv'
        result_df.to_csv(os.path.join(path, filename), encoding='utf-8', index=False)

print("All tests completed.")
