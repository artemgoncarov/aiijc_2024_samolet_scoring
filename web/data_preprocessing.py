import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from catboost import CatBoostClassifier
import joblib
import os
from feature_engine.imputation import MeanMedianImputer
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model import Model
from timeseries import TimeSeries
import warnings
warnings.filterwarnings('ignore')


model1 = Model()
model1 = model1.load_model('final/model_rf.joblib', 'final/model_cb.cbm')
model2 = Model()
model2 = model2.load_model('final/model_rf1.joblib', 'final/model_cb1.cbm')
model3 = Model()
model3 = model3.load_model('final/model_rf2.joblib', 'final/model_cb2.cbm')
simple_model = joblib.load('final/simple_model.joblib')
tsmodel = TimeSeries().load_model('final/timeseries.joblib')
print('MODELS WERE LOADED')

X = pd.read_csv('data/train_X.csv')
y = pd.read_csv('data/train_y.csv')
train = X.merge(y, on=['contract_id', 'report_date'])
graph = pd.read_csv('data/graph.csv').drop(columns='Unnamed: 0').rename(columns={'contractor_id1': 'contractor_id'})
full_graph = graph.pivot_table(index='contractor_id', columns='contractor_id2', values='Distance').fillna(0).reset_index()

connection_count = graph.groupby('contractor_id').size().to_dict()
mean_distance = graph.groupby('contractor_id').Distance.mean().to_dict()
min_distance = graph.groupby('contractor_id').Distance.min().to_dict()
max_distance = graph.groupby('contractor_id').Distance.max().to_dict()
var_distance = graph.groupby('contractor_id').Distance.var().to_dict()
median_distance = graph.groupby('contractor_id').Distance.median().to_dict()
std_distance = graph.groupby('contractor_id').Distance.std().to_dict()
count_distance = graph.groupby('contractor_id').Distance.count().to_dict()

copy_train = train.copy()
copy_train['default6'] = y.default6

to_drop, features = [], []
for i in copy_train.isnull().sum().items():
    if i[-1] > len(copy_train) * 0.7:        
        to_drop.append(i[0])
    if 0 < i[-1] <= len(copy_train) * 0.7:       
        features.append(i[0])

copy_train = copy_train.drop(columns=to_drop)
imputer = MeanMedianImputer(imputation_method='median', variables=features)
imputer.fit(copy_train[features])

dd = {}
contractors2 = list(set(copy_train.contractor_id))

for index, row in graph.iterrows():
    if row.Distance < 50000:
        if row.contractor_id not in dd:
            dd[row.contractor_id] = []
        if row.contractor_id2 in contractors2:
            dd[row.contractor_id].append((row.contractor_id2, row.Distance))

for j in dd:
    dd[j] = [i[0] for i in sorted(dd[j], key=lambda x: x[1])][:1]

def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 0  # Зима
    elif month in [3, 4, 5]:
        return 1  # Весна
    elif month in [6, 7, 8]:
        return 2  # Лето
    else:
        return 3  # Осень


def base_features(data: pd.DataFrame) -> pd.DataFrame:
    """A function for creating additional features based on dates,
    contract current and init sub, as well as other financial attributes.
    """
    data = data.copy()
    data['season'] = pd.to_datetime(data['contract_date']).apply(get_season)  # сезон подписания контракта
    data['month'] = pd.to_datetime(data['contract_date']).dt.month.apply(int)  # месяц подписания контракта
    data['day'] = pd.to_datetime(data['contract_date']).dt.day.apply(int)  # день подписания контракта
    data['day_of_week'] = pd.to_datetime(data['contract_date']).dt.dayofweek.apply(
        int)  # день недели подписания контракта
    data['year'] = pd.to_datetime(data['contract_date']).dt.year.apply(int)  # год подписания контракта
    data['report_month'] = pd.to_datetime(data['report_date']).dt.month.apply(int)  # месяц среза
    data['report_day'] = pd.to_datetime(data['report_date']).dt.day.apply(int)  # день среза
    data['contract_duration'] = (pd.to_datetime(data['report_date']) - pd.to_datetime(
        data['contract_date'])).dt.days  # длительность контракта
    data['time'] = pd.to_datetime(data['contract_date']).astype(int) / 10 ** 11  # datetime -> timestamp
    data['report_time'] = pd.to_datetime(data['report_date']).astype(int) / 10 ** 11  # datetime -> timestamp
    data['total_claims_last_12_months'] = data['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M'] + data[
        'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12M']  # сумма всех сборов за 12 месяцев
    data['total_claims_last_24_months'] = data['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M'] + data[
        'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_24M']  # сумма всех сборов за 24 месяца
    data['Income > Expenses'] = data['agg_FinanceAndTaxesFTS__g_contractor__Income__last__ALL_TIME'] > data[
        'agg_FinanceAndTaxesFTS__g_contractor__Expenses__last__ALL_TIME']  # доходы > расходы?
    data['Income > Expenses'] = data['Income > Expenses'].map(int)
    data['Income > Taxes + Expenses'] = data['agg_FinanceAndTaxesFTS__g_contractor__Income__last__ALL_TIME'] > (
                data['agg_FinanceAndTaxesFTS__g_contractor__TaxesSum__last__ALL_TIME'] + data[
            'agg_FinanceAndTaxesFTS__g_contractor__Expenses__last__ALL_TIME'])  # доходы > расходы + налоги?
    data['Income > Taxes + Expenses'] = data['Income > Taxes + Expenses'].map(int)
    data['credits'] = data['agg_spark_extended_report__g_contractor__CreditLimitSum__last__ALL_TIME'].round(
        2)  # округляем признак с кредитами
    data['tax'] = data['agg_FinanceAndTaxesFTS__g_contractor__TaxArrearsSum__last__ALL_TIME'] + data[
        'agg_FinanceAndTaxesFTS__g_contractor__TaxPenaltiesSum__last__ALL_TIME'] + data[
                      'agg_FinanceAndTaxesFTS__g_contractor__TaxesSum__last__ALL_TIME']  # сумма всех налогов
    data['contract_current_sum_mean_3M'] = data.groupby('contract_id')['contract_current_sum'].rolling(window=3,
                                                                                                       min_periods=1).mean().reset_index(
        0, drop=True)  # средняя сумма за 3 месяца по contract_id
    data['contract_sum_change_ratio'] = data['contract_current_sum'] / data['contract_init_sum']
    data['mean_weekly_abs_price_change'] = data[
        [f'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__last__ALL_TIME',
         f'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__mean__ALL_TIME']].mean(axis=1)
    data['arbitration_cases_12M'] = data[f'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M'] + \
                                    data[f'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12M']

    return data

def sums_features(data: pd.DataFrame) -> pd.DataFrame:
    """A function for creating additional features based on subtracting some features from others,
    namely, indicators for different periods of time. Thus, we create a feature increment of the indicator.
    Also, for each indicator, a mean and change attribute is created compared to the average and initial indicator.
    """

    data = data.copy()

    data['c1'] = data['agg_cec_requests__g_contract__request_id__all__count__2W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__1W']
    data['c2'] = data['agg_cec_requests__g_contract__request_id__all__count__3W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__2W']
    data['c3'] = data['agg_cec_requests__g_contract__request_id__all__count__4W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__3W']
    data['c4'] = data['agg_cec_requests__g_contract__request_id__all__count__5W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__4W']
    data['c5'] = data['agg_cec_requests__g_contract__request_id__all__count__6W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__5W']
    data['c6'] = data['agg_cec_requests__g_contract__request_id__all__count__7W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__6W']
    data['c7'] = data['agg_cec_requests__g_contract__request_id__all__count__8W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__7W']
    data['c8'] = data['agg_cec_requests__g_contract__request_id__all__count__12W'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__8W']
    data['c9'] = data['agg_cec_requests__g_contract__request_id__all__count__ALL_TIME'] - data[
        'agg_cec_requests__g_contract__request_id__all__count__12W']
    data['c_mean'] = (data['c1'] + data['c2'] + data['c3'] + data['c4'] + data['c5'] + data['c6'] + data['c7'] + data[
        'c8'] + data['c9']) / 9
    data['c_change'] = data['c_mean'] / data['c1']

    data['s1'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__2W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__1W']
    data['s2'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__3W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__2W']
    data['s3'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__4W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__3W']
    data['s4'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__5W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__4W']
    data['s5'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__6W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__5W']
    data['s6'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__7W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__6W']
    data['s7'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__8W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__7W']
    data['s8'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__12W'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__8W']
    data['s9'] = data['agg_cec_requests__g_contract__total_sum_accepted__all__sum__ALL_TIME'] - data[
        'agg_cec_requests__g_contract__total_sum_accepted__all__sum__12W']
    data['s_mean'] = (data['s1'] + data['s2'] + data['s3'] + data['s4'] + data['s5'] + data['s6'] + data['s7'] + data[
        's8'] + data['s9']) / 9
    data['s_change'] = data['s_mean'] / data['s1']

    data['m1'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__2M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__1M']
    data['m2'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__3M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__2M']
    data['m3'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__4M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__3M']
    data['m4'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__5M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__4M']
    data['m5'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__6M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__5M']
    data['m6'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__7M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__6M']
    data['m7'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__8M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__7M']
    data['m8'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__12M'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__8M']
    data['m9'] = data['agg_cec_requests__g_contract__time_btw_requests__all__mean__ALL_TIME'] - data[
        'agg_cec_requests__g_contract__time_btw_requests__all__mean__12M']
    data['m_mean'] = (data['m1'] + data['m2'] + data['m3'] + data['m4'] + data['m5'] + data['m6'] + data['m7'] + data[
        'm8'] + data['m9']) / 9
    data['m_change'] = data['m_mean'] / data['m1']

    data['d1'] = data['agg_payments__g_contract__sum__all__countDistinct__2W'] - data[
        'agg_payments__g_contract__sum__all__countDistinct__1W']
    data['d2'] = data['agg_payments__g_contract__sum__all__countDistinct__4W'] - data[
        'agg_payments__g_contract__sum__all__countDistinct__2W']
    data['d3'] = data['agg_payments__g_contract__sum__all__countDistinct__8W'] - data[
        'agg_payments__g_contract__sum__all__countDistinct__4W']
    data['d4'] = data['agg_payments__g_contract__sum__all__countDistinct__12W'] - data[
        'agg_payments__g_contract__sum__all__countDistinct__8W']
    data['d5'] = data['agg_payments__g_contract__sum__all__countDistinct__ALL_TIME'] - data[
        'agg_payments__g_contract__sum__all__countDistinct__12W']
    data['d_mean'] = (data['d1'] + data['d2'] + data['d3'] + data['d4'] + data['d5']) / 5
    data['d_change'] = data['d_mean'] / data['d1']

    data['a1'] = data['agg_payments__g_contract__sum__all__sum__2W'] - data[
        'agg_payments__g_contract__sum__all__sum__1W']
    data['a2'] = data['agg_payments__g_contract__sum__all__sum__4W'] - data[
        'agg_payments__g_contract__sum__all__sum__2W']
    data['a3'] = data['agg_payments__g_contract__sum__all__sum__8W'] - data[
        'agg_payments__g_contract__sum__all__sum__4W']
    data['a4'] = data['agg_payments__g_contract__sum__all__sum__12W'] - data[
        'agg_payments__g_contract__sum__all__sum__8W']
    data['a5'] = data['agg_payments__g_contract__sum__all__sum__ALL_TIME'] - data[
        'agg_payments__g_contract__sum__all__sum__12W']
    data['a_mean'] = (data['a1'] + data['a2'] + data['a3'] + data['a4'] + data['a5']) / 5
    data['a_change'] = data['a_mean'] / data['a1']

    data['ac1'] = data['agg_ks2__g_contract__id__all__count__2W'] - data['agg_ks2__g_contract__id__all__count__1W']
    data['ac2'] = data['agg_ks2__g_contract__id__all__count__4W'] - data['agg_ks2__g_contract__id__all__count__2W']
    data['ac3'] = data['agg_ks2__g_contract__id__all__count__8W'] - data['agg_ks2__g_contract__id__all__count__4W']
    data['ac4'] = data['agg_ks2__g_contract__id__all__count__12W'] - data['agg_ks2__g_contract__id__all__count__8W']
    data['ac5'] = data['agg_ks2__g_contract__id__all__count__ALL_TIME'] - data[
        'agg_ks2__g_contract__id__all__count__12W']
    data['ac_mean'] = (data['ac1'] + data['ac2'] + data['ac3'] + data['ac4'] + data['ac5']) / 5
    data['ac_change'] = data['ac_mean'] / data['ac1']

    data['as1'] = data['agg_ks2__g_contract__total_sum__all__sum__2W'] - data[
        'agg_ks2__g_contract__total_sum__all__sum__1W']
    data['as2'] = data['agg_ks2__g_contract__total_sum__all__sum__4W'] - data[
        'agg_ks2__g_contract__total_sum__all__sum__2W']
    data['as3'] = data['agg_ks2__g_contract__total_sum__all__sum__8W'] - data[
        'agg_ks2__g_contract__total_sum__all__sum__4W']
    data['as4'] = data['agg_ks2__g_contract__total_sum__all__sum__12W'] - data[
        'agg_ks2__g_contract__total_sum__all__sum__8W']
    data['as5'] = data['agg_ks2__g_contract__total_sum__all__sum__ALL_TIME'] - data[
        'agg_ks2__g_contract__total_sum__all__sum__12W']
    data['as_mean'] = (data['as1'] + data['as2'] + data['as3'] + data['as4'] + data['as5']) / 5
    data['as_change'] = data['as_mean'] / data['as1']

    data['w1'] = data['agg_spass_applications__g_contract__appl_count_week__mean__2W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__1W']
    data['w2'] = data['agg_spass_applications__g_contract__appl_count_week__mean__3W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__2W']
    data['w3'] = data['agg_spass_applications__g_contract__appl_count_week__mean__4W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__3W']
    data['w4'] = data['agg_spass_applications__g_contract__appl_count_week__mean__5W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__4W']
    data['w5'] = data['agg_spass_applications__g_contract__appl_count_week__mean__6W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__5W']
    data['w6'] = data['agg_spass_applications__g_contract__appl_count_week__mean__8W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__6W']
    data['w7'] = data['agg_spass_applications__g_contract__appl_count_week__mean__12W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__8W']
    data['w8'] = data['agg_spass_applications__g_contract__appl_count_week__mean__26W'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__12W']
    data['w9'] = data['agg_spass_applications__g_contract__appl_count_week__mean__ALL_TIME'] - data[
        'agg_spass_applications__g_contract__appl_count_week__mean__26W']
    data['w_mean'] = (data['w1'] + data['w2'] + data['w3'] + data['w4'] + data['w5'] + data['w6'] + data['w7'] + data[
        'w8'] + data['w9']) / 9
    data['w_change'] = data['w_mean'] / data['w1']

    data['f1'] = data['agg_workers__g_contract__fact_workers__all__mean__2W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__1W']
    data['f2'] = data['agg_workers__g_contract__fact_workers__all__mean__3W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__2W']
    data['f3'] = data['agg_workers__g_contract__fact_workers__all__mean__4W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__3W']
    data['f4'] = data['agg_workers__g_contract__fact_workers__all__mean__5W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__4W']
    data['f5'] = data['agg_workers__g_contract__fact_workers__all__mean__6W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__5W']
    data['f6'] = data['agg_workers__g_contract__fact_workers__all__mean__8W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__6W']
    data['f7'] = data['agg_workers__g_contract__fact_workers__all__mean__12W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__8W']
    data['f8'] = data['agg_workers__g_contract__fact_workers__all__mean__26W'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__12W']
    data['f9'] = data['agg_workers__g_contract__fact_workers__all__mean__ALL_TIME'] - data[
        'agg_workers__g_contract__fact_workers__all__mean__26W']
    data['f_mean'] = (data['f1'] + data['f2'] + data['f3'] + data['f4'] + data['f5'] + data['f6'] + data['f7'] + data[
        'f8'] + data['f9']) / 9
    data['f_change'] = data['f_mean'] / data['f1']

    data['o1'] = data['agg_materials__g_contract__order_id__countDistinct__2W'] - data[
        'agg_materials__g_contract__order_id__countDistinct__1W']
    data['o2'] = data['agg_materials__g_contract__order_id__countDistinct__4W'] - data[
        'agg_materials__g_contract__order_id__countDistinct__2W']
    data['o3'] = data['agg_materials__g_contract__order_id__countDistinct__8W'] - data[
        'agg_materials__g_contract__order_id__countDistinct__4W']
    data['o4'] = data['agg_materials__g_contract__order_id__countDistinct__12W'] - data[
        'agg_materials__g_contract__order_id__countDistinct__8W']
    data['o5'] = data['agg_materials__g_contract__order_id__countDistinct__ALL_TIME'] - data[
        'agg_materials__g_contract__order_id__countDistinct__12W']
    data['o_mean'] = (data['o1'] + data['o2'] + data['o3'] + data['o4'] + data['o5']) / 5
    data['o_change'] = data['o_mean'] / data['o1']

    data['i1'] = data['agg_sroomer__g_contractor__sroomer_id__count__6M'] - data[
        'agg_sroomer__g_contractor__sroomer_id__count__3M']
    data['i2'] = data['agg_sroomer__g_contractor__sroomer_id__count__12M'] - data[
        'agg_sroomer__g_contractor__sroomer_id__count__6M']
    data['i3'] = data['agg_sroomer__g_contractor__sroomer_id__count__ALL_TIME'] - data[
        'agg_sroomer__g_contractor__sroomer_id__count__12M']
    data['i_mean'] = (data['i1'] + data['i2'] + data['i3']) / 3
    data['i_change'] = data['i_mean'] / data['i1']

    data['ds1'] = data['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M'] - data[
        'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M']
    data['ds2'] = data['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_36M'] - data[
        'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M']
    data['ds3'] = data['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_48M'] - data[
        'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_36M']
    data['ds4'] = data['agg_ArbitrationCases__g_contractor__DefendantSum__sum__ALL_TIME'] - data[
        'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_48M']
    data['ds_mean'] = (data['ds1'] + data['ds2'] + data['ds3'] + data['ds4']) / 4
    data['ds_change'] = data['ds_mean'] / data['ds1']

    data['p1'] = data['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_24M'] - data[
        'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12M']
    data['p2'] = data['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_36M'] - data[
        'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_24M']
    data['p3'] = data['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_48M'] - data[
        'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_36M']
    data['p4'] = data['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__ALL_TIME'] - data[
        'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_48M']
    data['p_mean'] = (data['p1'] + data['p2'] + data['p3'] + data['p4']) / 4
    data['p_change'] = data['p_mean'] / data['p1']

    data['cd1'] = data['agg_tender_proposal__g_contractor__id__ALL__countDistinct__2W'] - data[
        'agg_tender_proposal__g_contractor__id__ALL__countDistinct__1W']
    data['cd2'] = data['agg_tender_proposal__g_contractor__id__ALL__countDistinct__4W'] - data[
        'agg_tender_proposal__g_contractor__id__ALL__countDistinct__2W']
    data['cd3'] = data['agg_tender_proposal__g_contractor__id__ALL__countDistinct__8W'] - data[
        'agg_tender_proposal__g_contractor__id__ALL__countDistinct__4W']
    data['cd4'] = data['agg_tender_proposal__g_contractor__id__ALL__countDistinct__12W'] - data[
        'agg_tender_proposal__g_contractor__id__ALL__countDistinct__8W']
    data['cd5'] = data['agg_tender_proposal__g_contractor__id__ALL__countDistinct__26W'] - data[
        'agg_tender_proposal__g_contractor__id__ALL__countDistinct__12W']
    data['cd6'] = data['agg_tender_proposal__g_contractor__id__ALL__countDistinct__52W'] - data[
        'agg_tender_proposal__g_contractor__id__ALL__countDistinct__26W']
    data['cd7'] = data['agg_tender_proposal__g_contractor__id__ALL__countDistinct__ALL_TIME'] - data[
        'agg_tender_proposal__g_contractor__id__ALL__countDistinct__52W']
    data['cd_mean'] = (data['cd1'] + data['cd2'] + data['cd3'] + data['cd4'] + data['cd5'] + data['cd6'] + data[
        'cd7']) / 7
    data['cd_change'] = data['cd_mean'] / data['cd1']

    return data


def default_percentage_radius(full_df: pd.DataFrame, graph: pd.DataFrame, r: int) -> pd.DataFrame:
    """
    A function for finding the default 6 percentage within a radius of R.
    For each contractor, other contractors are selected at a distance up to and including R.
    Next, the percentage of default6 is calculated by dividing the sum of all default6 by their number.
    """
    default6_proc = []

    for contractor_id in tqdm(list(full_df.contractor_id.unique())):
        l = list(graph[(graph.Distance <= r) & (graph.contractor_id == contractor_id)].contractor_id2.unique())
        cnt = 0
        default6 = 0

        for _, row in train.iterrows():
            if row.contractor_id in l or row.contractor_id == contractor_id:
                cnt += 1
                default6 += row.default6

        if cnt != 0:
            default6_proc.append({
                "contractor_id": contractor_id,
                "proc_default6": default6 / cnt
            })
        else:
            default6_proc.append({
                "contractor_id": contractor_id,
                "proc_default6": 0
            })

    return pd.DataFrame(default6_proc)

# агрегации над расстояниями

def distance_aggregations_features(df: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    """A function for creating aggregation features based on distances in the graph and df data. Returns pd.DataFrame with features"""
    data = []

    for i in list(df.contractor_id.unique()):
        try:
            a = graph[(graph.contractor_id == i) | (graph.contractor_id2 == i)].sort_values(by='Distance').head(5)
            l = []
            for _, row in a.iterrows():
                if row.contractor_id == i:
                    l.append(row.contractor_id2)
                else:
                    l.append(row.contractor_id)

            data.append({
                "contractor_id": i,
                "min_distance": min_distance[i],
                "max_distance": max_distance[i],
                "var_distance": var_distance[i],
                "median_distance": median_distance[i],
            })
        except:
            data.append({
                "contractor_id": i,
                "min_distance": 0,
                "max_distance": 0,
                "var_distance": 0,
                "median_distance": 0,
        })

    return pd.DataFrame(data)

def select_top5_neighbours(df: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    """A function for selecting the top 5 closest contractors. For each contractor,
    the 5 closest ones are selected and stored in a data frame.
    If there is no contractors, all columns add zero.
    """
    data = []

    for i in list(df.contractor_id.unique()):
        try:
            a = graph[(graph.contractor_id == i) | (graph.contractor_id2 == i)].sort_values(by='Distance').head(5)
            l = []
            for _, row in a.iterrows():
                if row.contractor_id == i:
                    l.append(row.contractor_id2)
                else:
                    l.append(row.contractor_id)

            data.append({
                "contractor_id": i,
                "mean_distance": mean_distance[i],
                "min_distance": min_distance[i],
                "max_distance": max_distance[i],
                "var_distance": var_distance[i],
                "median_distance": median_distance[i],
                "top1": l[0],
                "top2": l[1],
                "top3": l[2],
                "top4": l[3],
                "top5": l[4],
            })
        except:
            data.append({
                "contractor_id": i,
                "mean_distance": 0,
                "min_distance": 0,
                "max_distance": 0,
                "var_distance": 0,
                "median_distance": 0,
                "top1": 0,
                "top2": 0,
                "top3": 0,
                "top4": 0,
                "top5": 0,
            })

    return pd.DataFrame(data)

def top5_neighbours_features(df: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame: 
    d = []

    for _, row in tqdm(df.iterrows()):
        try:
            top1 = df[df.contractor_id == row.top1]
            top2 = df[df.contractor_id == row.top2]
            top3 = df[df.contractor_id == row.top3]
            top4 = df[df.contractor_id == row.top4]
            top5 = df[df.contractor_id == row.top5]

            top1init = top1.contract_init_sum.mean()
            top1curr = top1.contract_current_sum.mean()

            top2init = top2.contract_init_sum.mean()
            top2curr = top2.contract_current_sum.mean()

            top3init = top3.contract_init_sum.mean()
            top3curr = top3.contract_current_sum.mean()

            top4init = top4.contract_init_sum.mean()
            top4curr = top4.contract_current_sum.mean()

            top5init = top5.contract_init_sum.mean()
            top5curr = top5.contract_current_sum.mean()

            d.append({
                "top1init": top1init,
                "top1curr": top1curr,
                "top2init": top2init,
                "top2curr": top2curr,
                "top3init": top3init,
                "top3curr": top3curr,
                "top4init": top4init,
                "top4curr": top4curr,
                "top5init": top5init,
                "top5curr": top5curr
            })
        except:
            d.append({
                "top1init": 0,
                "top1curr": 0,
                "top2init": 0,
                "top2curr": 0,
                "top3init": 0,
                "top3curr": 0,
                "top4init": 0,
                "top4curr": 0,
                "top5init": 0,
                "top5curr": 0
            })

    return pd.DataFrame(d)

def simple_preprocess(test) -> pd.DataFrame:
    copy_test1 = test.copy()
    copy_test1 = copy_test1.drop(columns=['contract_id','project_id', 'building_id'])
    copy_test1[features] = imputer.transform(copy_test1[features])
    
    return copy_test1

def preprocess(test, copy_train,  path):
    copy_train = copy_train.copy()
    contractors3 = list(test.contractor_id)
    sum_all_sum = test.specialization_sum_agg_payments__g_contract__sum__all__sum__ALL_TIME

    train1 = train.copy()
    copy_test1 = simple_preprocess(test)
    print('SIMPLE IS DONE')

    train1 = base_features(train1)
    test = base_features(test)

    train1 = sums_features(train1)
    test = sums_features(test)

    print('FEATURES WERE CREATED')

    train1 = train1.drop(columns=['report_date', 'contract_date', 'contract_id'])  # delete dates и contract_id
    full_df = pd.concat(
        [train1.drop(columns='default6'), test.drop(columns=['report_date', 'contract_date', 'contract_id'])],
        ignore_index=True) # concating train and test

    full_df['specialization_id'] = full_df['specialization_id'].map(int)
    full_df['project_id'] = full_df['project_id'].map(int)
    full_df['building_id'] = full_df['building_id'].map(int)
    full_df['contractor_id'] = full_df['contractor_id'].map(int)

    # one-hot encoding for ID features
    specs = pd.get_dummies(full_df.specialization_id).map(int).rename(columns=dict(
        zip(full_df.specialization_id.unique().tolist(),
            [f'spec_{i}' for i in full_df.specialization_id.unique().tolist()])))
    projects = pd.get_dummies(full_df.project_id).map(int).rename(
        columns=dict(zip(full_df.project_id.unique(), [f'project_{i}' for i in full_df.project_id.unique().tolist()])))
    buildings = pd.get_dummies(full_df.building_id).map(int).rename(columns=dict(
        zip(full_df.building_id.unique().tolist(), [f'building_{i}' for i in full_df.building_id.unique().tolist()])))
    contractors = pd.get_dummies(full_df.contractor_id).map(int).rename(columns=dict(
        zip(full_df.contractor_id.unique().tolist(), [f'contractor_{i}' for i in full_df.contractor_id.unique().tolist()])))
    
    # join ids
    full_df = full_df.join(specs)
    full_df = full_df.join(projects)
    full_df = full_df.join(buildings)
    full_df = full_df.join(contractors)

    default6_proc = pd.read_csv('default6_proc (1).csv')

    distance_agg = distance_aggregations_features(full_df, graph)

    full_df = full_df.merge(distance_agg, on='contractor_id') # merge full_df and data

    full_df = full_df.merge(default6_proc, on='contractor_id') # merge default6_proc and data

    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(full_df)
    df_imputed = pd.DataFrame(df_imputed, columns=full_df.columns)

    full_df = df_imputed.copy() #[len(train):]

    top5 = select_top5_neighbours(full_df, graph)

    full_df = full_df.merge(top5, on='contractor_id').rename(columns={
        'median_distance_y': 'median_distance',
        'mean_distance_y': 'mean_distance',
        'min_distance_y': 'min_distance',
        'max_distance_y': 'max_distance',
        'var_distance_y': 'var_distance'
    })

    data = top5_neighbours_features(full_df, graph).fillna(0)

    full_df = pd.concat([full_df, data], axis=1)

    print('TOP 5 IS DONE')

    preds1 = model1.predict_proba(full_df[len(train):][model1.rf_model.feature_names_in_.tolist()])
    preds2 = model2.predict_proba(full_df[len(train):][model2.rf_model.feature_names_in_.tolist()])
    preds3 = model3.predict_proba(full_df[len(train):][model3.rf_model.feature_names_in_.tolist()])
    preds = preds1 * 0.5 + preds2 * 0.2 + preds3 * 0.3
    print('PREDICTED')

    copy_test = test[['contract_id', 'report_date']]
    copy_test['score'] = preds

    temp = full_df[len(train):].copy().reset_index(drop=True)

    temp['score_from_cross_val'] = preds # for stemming
    temp['c'] = range(len(test))
    temp['contractor_id'] = contractors3
    temp['score'] = preds

    s2 = tsmodel.time_series_predict(temp)[0]
    score = s2.score

    print('TIME SERIES')

    f = copy_test[['contract_id', 'report_date', 'score']]
    f['score'] = score

    copy_train = copy_train.drop(columns=['report_date', 'contract_date'])
    data = copy_train.groupby(['contractor_id'], as_index=False).mean()
    scores = []
    for i in tqdm(range(len(copy_test1))):
        pred = simple_model.predict_proba(copy_test1.iloc[[i], :][simple_model.feature_names_in_.tolist()])[:, 1][0]
        if copy_test1.iloc[i, :]['contractor_id'] in dd:
            simple_row = copy_test1[simple_model.feature_names_in_.tolist()].iloc[[i], :]
            contractor_mean_data = data[copy_train.contractor_id == dd[copy_test1.iloc[i, :]['contractor_id']][0]]

            row_without_contractor = simple_row.drop(columns=[column for column in simple_row.columns if 'contractor' in column])
            contractor_info = contractor_mean_data.filter(like='contractor')
            if len(contractor_info) == 0:
                scores.append(f.iloc[i][2])
            else:
                row_without_contractor[contractor_info.columns] = contractor_info.iloc[0]
                scores.append((simple_model.predict_proba(row_without_contractor[simple_model.feature_names_in_.tolist()])[:, 1][0] + f.iloc[i][2]) / 2)        
        else:
            scores.append(f.iloc[i][2])

    print('ALGORITHM')

    f.score = pd.Series(scores)
    temp['score'] = pd.Series(scores)
    temp.to_csv(path[:-4]+"©"+'.csv', index=False)
    f.to_csv(path, index=False)