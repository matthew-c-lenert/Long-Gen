import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.special import logit
from scipy.stats import pareto
from itertools import product
import sys
from operator import mul
from functools import reduce
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
# Functional form last
# Stationarity first
# Colinearity middle
# Sampling middle
# Measurment occaisons


def random_offset(value):
    if value<=0.0001:
        return(value+np.random.uniform(low=0.0005, high=0.05))
    elif value>=0.9999:
        return(value-np.random.uniform(low=0.0005, high=0.05))
    else:
        return(value)


def piecewise_sample(stationarity_change_points,is_high,index,max_index):
    high=stationarity_change_points[max_index]
    low=0
    if is_high:
        high=stationarity_change_points[index]
        if index>0:
            low=stationarity_change_points[index-1]
    else:
        low=stationarity_change_points[index]
        if index<max_index:
            high=stationarity_change_points[index+1]
    return(np.random.uniform(low=low, high=high))

def normalize_adjustment(feature_values):
    mod_vals=np.ceil(feature_values)
    vals2=(feature_values==2)*1.1
    vals3=(feature_values==3)*1.25
    vals4=(feature_values>3)*1.5
    return(vals2+vals3+vals4)



def get_sample_times(bucket,measurment_occaisons,feature_values,sampling_param=None,sampling_function=None):
    if bucket == "equal":
        return(np.linspace(0,0.99999,measurment_occaisons))
    elif bucket == "random":
        return(np.random.uniform(low=0.0, high=1.0, size=measurment_occaisons))
    elif bucket == "not-random":
        abnormal_x={}
        either=[]
        first=True
        for feature in feature_values:
            ab_x=np.abs(feature_values[feature])>=1
            if first:
                first=False
                either=ab_x
            else:
                either=either+ab_x
        neither=np.logical_not(either)

        precent_ab=np.sum(either)/measurment_occaisons
        step_size=((0.5+0.5*precent_ab)/(measurment_occaisons-1))*np.ones(measurment_occaisons)

        for feature in feature_values:
            abnormal_x[feature]=(step_size/normalize_adjustment(np.abs(feature_values[feature])))*either
            abnormal_x[feature]=np.insert(abnormal_x[feature],0,0)[0:-1]

        neither=np.insert(neither,0,False)[0:-1]
        return(np.cumsum(step_size*neither+np.minimum.reduce([abnormal_x[x] for x in abnormal_x])))
    elif bucket == "custom-no-features":
        return(sampling_function(measurment_occaisons))
    elif bucket == "custom-feature-values":
        return(sampling_function(measurment_occaisons,feature_values))


def split_between_cutpoints(times,entity_to_split,stationarity_change_points):
    stationarity_index=0
    i=0
    model_count=0
    return_dict={}
    items_to_sort=len(times)
    periods=len(stationarity_change_points)

    while i < items_to_sort:
        s_time=1
        if stationarity_index < periods:
            s_time=stationarity_change_points[stationarity_index]
        if times[i] < s_time:
            if model_count not in return_dict:
                return_dict[model_count]=[entity_to_split[i]]
            else:
                return_dict[model_count].append(entity_to_split[i])
            i+=1
        else:
            stationarity_index+=1
            model_count+=1
    return(return_dict)


def get_colinearity(bucket,count_vector):
    if bucket == "low-low":
        return((0.1,round(np.random.uniform(low=0.005, high=0.04),3),np.random.uniform(0.01,0.33)))
    elif bucket == "low-moderate":
        return((0.1,round(np.random.uniform(low=0.05, high=0.12),2),np.random.uniform(0.01,0.33)))
    elif bucket == "low-high":
        return((0.1,round(np.random.uniform(low=0.13, high=0.3),2),np.random.uniform(0.01,0.33)))
    elif bucket == "moderate-high":
        return((round(np.random.uniform(low=0.75, high=0.9),2),1.5,np.random.uniform(0.33,0.66)))
    elif bucket == "high-high":
        return((round(np.random.uniform(low=1.0, high=2.0),2),1.5,np.random.uniform(0.66,0.999)))


def get_relative_time(sorted_time_points):
    offset_time=np.insert(sorted_time_points,0,sorted_time_points[0])
    offset_time=np.delete(offset_time,-1)
    return(sorted_time_points-offset_time)


def get_stationarity_change_points(stationarity_count):
    return(np.random.uniform(low=0.0, high=1.0, size=stationarity_count))


def binary_y(arr_y):
    draw=np.random.uniform(0,1,arr_y.size)
    return((draw<=arr_y).astype(int))


class patient_model():

    def __init__(self,period_index,b_values,coefficient_values,link_fn,obs_error,times,first_obs_index,relative_time,period_features,period_extraneous_variables,stationarity_trend_bucket):
        self.b_values={}
        self.coefficient_values={}
        for b in b_values:
            self.b_values[b]=b_values[b]

        for coefficient in coefficient_values:
            self.coefficient_values[coefficient]=coefficient_values[coefficient][period_index]

        self.time_points=times
        self.num_obs=len(times)
        self.link_fn=link_fn
        self.obs_index=np.array(range(first_obs_index,first_obs_index+self.num_obs))
        self.features=period_features
        self.extraneous_variables=period_extraneous_variables
        self.obs_error=obs_error
        self.relative_time=relative_time
        self.stationarity_trend_bucket=stationarity_trend_bucket

    def generate_data(self):
        y=np.ones(self.num_obs)*self.coefficient_values["intercept"]
        y+=np.multiply(np.ones(self.num_obs)*self.coefficient_values["time"],self.time_points)
        if self.stationarity_trend_bucket=="quadratic":
            y+=np.multiply(np.ones(self.num_obs)*self.coefficient_values["trend-time"],np.sqrt(self.time_points))
        elif self.stationarity_trend_bucket=="seasonal":
            y+=np.multiply(np.ones(self.num_obs)*self.coefficient_values["trend-time"],np.cos(np.multiply(np.pi*10,self.time_points)))

        for feature in self.features:
            y+=self.coefficient_values[feature]*np.array(self.features[feature])

        for effect in self.b_values:
            if effect=="intercept":
                y+=np.ones(self.num_obs)*self.b_values[effect]
            elif effect=="time":
                y+=np.multiply(np.ones(self.num_obs)*self.b_values[effect],self.time_points)
            elif effect=="trend-time":
                if self.stationarity_trend_bucket=="quadratic":
                    y+=np.multiply(np.ones(self.num_obs)*self.b_values[effect],np.sqrt(self.time_points))
                elif self.stationarity_trend_bucket=="seasonal":
                    y+=np.multiply(np.ones(self.num_obs)*self.b_values[effect],np.cos(np.multiply(np.pi*10,self.time_points)))
            else:
                y+=self.b_values[eff]*self.features[effect]

        if self.link_fn=="identity":
            self.y=y+self.obs_error
        elif self.link_fn=="log":
            self.y=np.exp(y)+self.obs_error
        elif self.link_fn=="logit":
            self.y_prob=np.maximum(expit(y),self.obs_error)
            self.y=binary_y(self.y_prob)
        elif self.link_fn=="inverse":
            self.y=np.power(y,-1)+self.obs_error

    def export_data_frame(self):
        if self.link_fn=="logit":
            df={"obs_index":list(self.obs_index),"y":list(self.y),"y_prob":list(self.y_prob),"time":list(self.time_points),"relative_time":list(self.relative_time), "unobserved_error":list(self.obs_error)}
            for feature in self.features:
                df[feature]=self.features[feature]
            for variable in self.extraneous_variables:
                df[variable]=self.extraneous_variables[variable]
            return(df)
        else:
            df={"obs_index":list(self.obs_index),"y":list(self.y),"time":list(self.time_points),"relative_time":list(self.relative_time), "unobserved_error":list(self.obs_error)}
            for feature in self.features:
                df[feature]=self.features[feature]
            for variable in self.extraneous_variables:
                df[variable]=self.extraneous_variables[variable]
            return(df)


class patient:

    def __init__(self,pat_id,b_values,features,coefficient_values,extraneous_variables,colinearity,stationarity_change_points,measurements,sampling_bucket,link_fn,sigma_e,stationarity_trend_bucket,sampling_function=None):
        self.b_values=b_values
        self.id=pat_id
        self.coefficient_values=coefficient_values
        self.models=[]
        self.sigma_e=sigma_e
        self.stationarity_trend_bucket=stationarity_trend_bucket
        if measurements<1:
            self.measure_count=1
        else:
            self.measure_count=int(measurements)
        obs_error=[]
        if link_fn=="identity":
            obs_error=np.random.normal(0,self.sigma_e,self.measure_count)
        elif link_fn=="log":
            obs_error=np.random.poisson(self.sigma_e,self.measure_count)
        elif link_fn=="logit":
            new_sigma_e=(1.0-np.sqrt(1.0-4.0*self.sigma_e**(3)/self.measure_count))/2.0
            obs_error=np.random.binomial(self.measure_count,new_sigma_e,self.measure_count)
        elif link_fn=="inverse":
            obs_error=np.random.gamma(1,np.sqrt(self.sigma_e)/self.measure_count,self.measure_count)
        periods=len(stationarity_change_points)+1

        feature_values={}
        extraneous_variable_values={}
        b_factor=0

        if len(self.b_values) > 0:
            if "intercept" in self.b_values:
                b_factor=self.b_values["intercept"]
            elif "time" in self.b_values:
                b_factor=self.b_values["time"]
            elif "trend-time" in self.b_values:
                b_factor=self.b_values["trend-time"]
            else:
                for b in b_values:
                    b_factor=self.b_values[b]
                    break

        if (sampling_bucket=="not-random") | (sampling_bucket=="custom-feature-values"):
            total_length=len(features)+len(extraneous_variables)
            x_cov_matrix=np.ones((total_length,total_length))
            np.fill_diagonal(x_cov_matrix,colinearity[2])
            x=np.random.multivariate_normal(tuple(np.ones(total_length)*b_factor), x_cov_matrix, self.measure_count)
            index=0
            for feature in features:
                feature_values[feature]=x[:,index]
                index+=1
            index=0
            for variable in extraneous_variables:
                extraneous_variable_values[variable]=x[:,index]
                index+=1
        time_points=get_sample_times(sampling_bucket,self.measure_count,feature_values,sampling_function)
        time_points=np.sort(time_points)

        if (sampling_bucket!="not-random") & (sampling_bucket!="custom-feature-values"):
            kernel=(1.0+np.abs(b_factor)) * Matern(length_scale=colinearity[0], length_scale_bounds=(1e-5, 1e5), nu=colinearity[1])
            gp = GaussianProcessRegressor(kernel=kernel)
            y_samples = gp.sample_y(time_points[:, np.newaxis],len(features))
            index=0
            for feature in features:
                feature_values[feature]=y_samples.T[index]
                index+=1
            index=0
            for variable in extraneous_variables:
                extraneous_variable_values[variable]=y_samples.T[index]
                index+=1

        relative_time=get_relative_time(time_points)
        sorted_times=split_between_cutpoints(time_points,time_points,stationarity_change_points)
        relative_time=split_between_cutpoints(time_points,relative_time,stationarity_change_points)
        for feature in feature_values:
            feature_values[feature]=split_between_cutpoints(time_points,feature_values[feature],stationarity_change_points)
        for variable in extraneous_variable_values:
            extraneous_variable_values[variable]=split_between_cutpoints(time_points,extraneous_variable_values[variable],stationarity_change_points)
        obs_error=split_between_cutpoints(time_points,obs_error,stationarity_change_points)
        period_index=0
        obs_index=1
        for key in sorted_times:
            period_features={}
            period_extraneous_variables={}
            for feature in feature_values:
                period_features[feature]=feature_values[feature][key]

            for variable in extraneous_variable_values:
                period_extraneous_variables[variable]=extraneous_variable_values[variable][key]

            self.models.append(patient_model(period_index,self.b_values,self.coefficient_values,link_fn,np.array(obs_error[key]),sorted_times[key],obs_index,relative_time[key],period_features,period_extraneous_variables,stationarity_trend_bucket))
            period_index+=1
            obs_index+=len(sorted_times[key])

    def export_to_data_frame(self):
        first=True
        return_frame={}
        for model in self.models:
            model.generate_data()
            if first:
                return_frame=model.export_data_frame()
                first=False
            else:
                new_data=model.export_data_frame()
                for key in return_frame:
                    return_frame[key].extend(new_data[key])
        if self.measure_count>0:
            return_frame['pat_id']=list(np.ones(len(return_frame["time"]))*self.id)
        return(return_frame)


class long_data_set:

    def __init__(self,n=2000,num_measurements=25,collinearity_bucket="low-low",trend_bucket="linear",sampling_bucket="random",sampling_function=None,b_colin=0.13,beta_var=1,b_var=1,time_importance_factor=3,sigma_e=0.05,num_features=2,num_extraneous_variables=0,link_fn="identity",num_piecewise_breaks=0,random_effects=["intercept","time","trend-time"],coefficient_values={},time_breaks=[]):
        self.num_of_patients=n
        self.num_measurements=num_measurements
        self.colinearity_bucket=collinearity_bucket
        self.stationarity_trend_bucket=trend_bucket
        self.num_piecewise_breaks=num_piecewise_breaks
        self.sampling_bucket=sampling_bucket
        self.link_fn=link_fn
        self.inflation_factor=time_importance_factor
        self.beta_var=beta_var
        self.b_var=b_var
        self.b_colin=b_colin
        self.sigma_e=sigma_e
        self.sampling_function=sampling_function
        ###############################
        self.features=[]
        for i in range(num_features):
            self.features.append("x"+str(i+1))

        self.extraneous_variables=[]
        for i in range(num_extraneous_variables):
            self.extraneous_variables.append("ext_"+str(i+1))

        self.random_effects=random_effects
        self.coefficient_values=coefficient_values
        self.time_breaks=time_breaks
        ###############################

    def create_data_set(self):
        if len(self.time_breaks)>0:
            if len(self.time_breaks)!=self.num_piecewise_breaks:
                raise ValueError('Number of specific time_breaks do not match num_piecewise_breaks')
            else:
                self.change_points=np.sort(self.time_breaks)
        else:
            self.change_points=np.sort(get_stationarity_change_points(self.num_piecewise_breaks))
        measures=pareto.rvs(3.5, loc=self.num_measurements-self.num_measurements/10, scale=2.5, size=self.num_of_patients, random_state=None)
        measures=np.sort(np.round(measures,0))

        ro_x=get_colinearity(self.colinearity_bucket,self.num_of_patients)
        b_cov_matrix=np.zeros((len(self.random_effects),len(self.random_effects)))
        b_cov_matrix.fill(np.sqrt(self.b_var)*self.b_colin)
        np.fill_diagonal(b_cov_matrix,self.b_var)
        b=np.random.multivariate_normal(tuple(np.zeros(len(self.random_effects))), b_cov_matrix, self.num_of_patients)
        self.b_dict={}
        b_index=0
        for effect in self.random_effects:
            self.b_dict[effect]=b[:,b_index]
            b_index+=1
        b_df=pd.DataFrame(self.b_dict)
        if len(self.random_effects) > 0:
            if "intercept" in self.b_dict:
                b_df['sort_col']=b_df["intercept"].abs()
            elif "time" in self.b_dict:
                b_df['sort_col']=b_df["time"].abs()
            elif "trend-time" in self.b_dict:
                b_df['sort_col']=b_df["trend-time"].abs()
            else:
                sort_col=b_df.columns.values[0]
                b_df['sort_col']=b_df[sort_col].abs()
            b_df=b_df.sort_values(by=['sort_col'])
            b_df=b_df.reset_index(drop=True)
        for effect in self.random_effects:
            self.b_dict[effect]=b_df[effect].values

        long_data=[]
        first=True
        periods=len(self.change_points)+1

        if len(self.coefficient_values)==0:
            self.coefficient_values={"intercept":np.zeros(periods),"time":np.zeros(periods),"trend-time":np.zeros(periods)}
            for feature in self.features:
                self.coefficient_values[feature]=np.zeros(periods)

            for i in range(periods):
                for feature in self.coefficient_values:
                    if feature=="time":
                        self.coefficient_values[feature][i]=np.random.normal(self.coefficient_values[feature][i-1],self.inflation_factor*self.beta_var)
                    elif feature=="trend-time":
                        self.coefficient_values[feature][i]=np.random.normal(self.coefficient_values[feature][i-1],self.inflation_factor*self.beta_var)
                    else:
                        self.coefficient_values[feature][i]=np.random.normal(self.coefficient_values[feature][i-1],self.beta_var)

        for p_id in range(self.num_of_patients):
            b_values={}
            for b in self.b_dict:
                b_values[b]=self.b_dict[b][p_id]
            pat=patient(p_id,b_values,self.features,self.coefficient_values,self.extraneous_variables,ro_x,self.change_points,measures[p_id],self.sampling_bucket,self.link_fn,self.sigma_e,self.stationarity_trend_bucket,self.sampling_function)
            if first:
                first=False
                long_data=pat.export_to_data_frame()
            else:
                new_data=pat.export_to_data_frame()
                for key in long_data:
                    long_data[key].extend(new_data[key])
        self.data_frame=pd.DataFrame(long_data)

    def export_to_csv(self,path_name,file_name):
        self.data_frame.to_csv(path_name+"data_"+file_name+".csv")
        param_values={"piecewise_shifts":np.append(self.change_points,[1]),"cons_":self.coefficient_values["intercept"],"time_linear":self.coefficient_values["time"]}
        if self.stationarity_trend_bucket=="quadratic":
            param_values["cos_time"]=self.coefficient_values["trend-time"]
        elif self.stationarity_trend_bucket=="seasonal":
            param_values["sqrt_time"]=self.coefficient_values["trend-time"]
        for feature in self.features:
            param_values[feature]=self.coefficient_values[feature]

        pd.DataFrame(param_values).to_csv(path_name+"params_"+file_name+".csv")

    def transform_variable_feature(self,column_names,transformation_function):
        comparison_change_points=[0]+list(self.change_points)+[1]
        if "time" in column_names:
            self.change_points=transformation_function(self.change_points)

        new_y=[]
        new_prob_y=[]
        new_x=[]
        self.data_frame=self.data_frame.sort_values(by=['time'])
        if self.link_fn=="logit":
            self.data_frame["new_y"]=logit(self.data_frame["y_prob"])

        elif self.link_fn=="log":
            self.data_frame["new_y"]=np.log(self.data_frame["y"]-self.data_frame["unobserved_error"])

        elif self.link_fn=="inverse":
            self.data_frame["new_y"]=np.power(self.data_frame["y"]-self.data_frame["unobserved_error"],-1)

        else:
            self.data_frame["new_y"]=self.data_frame["y"]-self.data_frame["unobserved_error"]

        for column in column_names:
            for period_index in range(1,len(comparison_change_points)):
                period_data_frame=self.data_frame[(self.data_frame['time'] > comparison_change_points[period_index-1]) & (self.data_frame['time'] <= comparison_change_points[period_index])]
                period_y=[]
                period_x=[]

                period_x=transformation_function(period_data_frame[column])

                if column in self.features:
                    period_y=period_data_frame["new_y"]+(self.coefficient_values[column][period_index-1]*(period_x-period_data_frame[column]))

                if column in self.b_dict:
                    period_y=period_data_frame["new_y"]+np.multiply(self.b_dict[column][period_index-1],(period_x-period_data_frame[column]))

                if column == "time":
                    if (self.stationarity_trend_bucket=="quadratic") | (self.stationarity_trend_bucket=="seasonal"):
                        period_y=period_data_frame["new_y"]+(self.coefficient_values["trend-time"][period_index-1]*(period_x-period_data_frame[column]))
                        if "trend-time" in self.b_dict:
                            period_y=period_data_frame["new_y"]+np.multiply(self.b_dict["trend-time"][period_index-1],(period_x-period_data_frame[column]))

                if len(period_y)>0:
                    if period_index==1:
                        new_x=period_x
                        new_y=period_y
                    else:
                        new_x=np.concatenate([new_x,period_x])
                        new_y=np.concatenate([new_y,period_y])

            self.data_frame["new_"+column]=new_x
            self.data_frame["new_y"]=new_y

        if self.link_fn=="logit":
            self.data_frame["new_y_prob"]=expit(self.data_frame["new_y"])
            self.data_frame["new_y"]=binary_y(self.data_frame["new_y_prob"])

        elif self.link_fn=="log":
            self.data_frame["new_y"]=np.exp(self.data_frame["new_y"])+self.data_frame["unobserved_error"]

        elif self.link_fn=="inverse":
            self.data_frame["new_y"]=np.power(self.data_frame["new_y"],-1)+period_data_frame["unobserved_error"]

        else:
            self.data_frame["new_y"]=self.data_frame["new_y"]+self.data_frame["unobserved_error"]

