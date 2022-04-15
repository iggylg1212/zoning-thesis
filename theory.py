from matplotlib.pyplot import axis
import numpy as np
from tqdm import tqdm
from itertools import permutations
import pandas as pd
import math
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.linalg import toeplitz

def in_gov(j,c_i,b_i):
    if boundaries[j-1][0] <= (c_i+b_i)/2 <= boundaries[j-1][1]:
        return 1 
    else:
        return 0

def in_city(c_i,b_i):
    if (c_i+b_i)/2<R & (c_i+b_i)/2>0: 
        return 1
    else:
        return 0

def out_city(c_i,b_i):
    if (c_i+b_i)/2>=R or (c_i+b_i)/2<=0: 
        return 1
    else:
        return 0

def capital(j, p_j, s_j, alpha_j, gamma_j, tau_j):
    x = (1/3)*p_j*(boundaries[j-1][1]**3-boundaries[j-1][0]**3)+(1/2)*s_j*(boundaries[j-1][1]**2-boundaries[j-1][0]**2)
    if x <=0:
        return 0
    if 0 < x <= (1/2)*((alpha_j/(np.pi*(boundaries[j-1][1]**2-boundaries[j-1][0]**2)))**((1-d)/d))*(1/d)*(1/(1-gamma_j))*(boundaries[j-1][1]**2-boundaries[j-1][0]**2)*(k+tau_j):
        return ((1/d)*(1/(1-gamma_j))*(1/((p_j/3)*(boundaries[j-1][1]**3-boundaries[j-1][0]**3)+(s_j/2)*(boundaries[j-1][1]**2-boundaries[j-1][0]**2)))*(1/2)*(boundaries[j-1][1]**2-boundaries[j-1][0]**2)*(k+tau_j))**(1/(d-1))
    if (1/2)*((alpha_j/(np.pi*(boundaries[j-1][1]**2-boundaries[j-1][0]**2)))**((1-d)/d))*(1/d)*(1/(1-gamma_j))*(boundaries[j-1][1]**2-boundaries[j-1][0]**2)*(k+tau_j)<x:
        return (alpha_j*(1/np.pi)*(1/(boundaries[j-1][1]**2-boundaries[j-1][0]**2)))**(1/d)

## Exogenous Parameters
np.random.seed(0)

N = 5
U = 1
R = (3/np.pi)**(1/3)

d = .5
k = 300

high_income = 100000
low_income = 50000

housing_error_param = 40
utility_error_param = .25
pg_error_param = .25

housing_pref = np.random.beta(housing_error_param,housing_error_param,N)
utility_pref = np.random.lognormal(0,utility_error_param,N)
pg_pref = np.random.lognormal(0,pg_error_param,N)

perc_high = .3
incomes = [high_income for x in range(int(perc_high*N))]
incomes.extend([low_income for x in range(int((1-perc_high)*N)+1)])

alpha_list = [172.1567434,292.3063477,353.197233]
beta_list = [3.256149152,3.551292258,4.21479788]
tau_list = [366.12148,494.2652862,583.2918017]
gamma_list = [0.324316548,0.578694524]

pol_list = []
alphas = permutations(alpha_list,2)
for alphin in alphas:
    betas = permutations(beta_list,2)
    for betin in betas:
        taus = permutations(tau_list,2)
        for tauin in taus:
            gammas = permutations(gamma_list,2)
            for gammin in gammas:
                pol_list.append((alphin,betin,tauin,gammin))

governmental_analysis = pd.DataFrame()
boundary_conditions = [.45,.5,.55,.6,.65]
for boundarycondition in boundary_conditions:
    boundary = boundarycondition
    boundaries = [(0,boundary), (boundary, R)]
    J = len(boundaries)
    ## Generate Pairs
    pairs = list(range(1,N+1))
    options_df = pd.DataFrame(columns=['gov1','gov2','out'])
    for i in range(0,N+1):
        gov1 =  permutations(pairs, i)
        for perm in gov1:
            cands_gov2 = [x for x in pairs if x not in perm]
            for ii in range(0,N+1-i):
                gov2 = permutations(cands_gov2,ii)
                for perm2 in gov2:
                    options_df = pd.concat([options_df,pd.DataFrame.from_dict({'gov1':[perm],'gov2':[perm2],'out':[tuple([x for x in pairs if x not in perm and x not in perm2])]})])
    options_df = options_df.reset_index(drop=True)
    options_df['coords'] = ''
    for index,row in options_df.iterrows():
        coords = list(range(1,N+1))
        gov1 = row['gov1']
        for x in range(len(gov1)):
            slice1 = boundary/len(gov1)
            if x==0:
                numb = gov1[x]
                coords[numb-1] = (0, slice1)
            else:
                numb = gov1[x]
                coords[numb-1] = (x*slice1,(x+1)*slice1)
        gov2 = row['gov2']
        for x in range(len(gov2)):
            slice2 = (R-boundary)/len(gov2)
            if x==0:
                numb = gov2[x]
                coords[numb-1] = (boundary, boundary+slice2)
            else:
                numb = gov2[x]
                coords[numb-1] = (boundary+x*slice2, boundary+(x+1)*slice2)
        out = row['out']
        for x in range(len(out)):
            numb = out[x]
            coords[numb-1] = (R,R)    
        final_coords = []
        for i in coords:
            for ii in i:
                final_coords.append(ii)
        final_coords = np.reshape(np.asarray(final_coords),(1,2*N))
        options_df.iloc[index,3] = final_coords
        
    options_df['in_gov1'] = options_df['gov1'].apply(lambda x: len(x))
    options_df['in_gov2'] = options_df['gov2'].apply(lambda x: len(x))
    options_df['in_city'] = options_df.apply(lambda x: x['in_gov1']+x['in_gov2'],axis=1)

    for one in tqdm(range(len(pol_list))):
        pols = pol_list[one]

        alpha = pols[0]
        beta = pols[1]
        tau = pols[2]
        gamma = pols[3]

        def demand_objective(x):
            obj = 0
            for index in gov_list:
                income = incomes[index-1]
                c_i = row['coords'][0][2*(index-1)+1]
                b_i = row['coords'][0][2*(index-1)]
                cap= capital(j-1,x[0],x[1],alpha[j-1],gamma[j-1],tau[j-1])**d
                arg = 1-2*np.pi*(1/income)*cap*(x[0]*(1/3)*(c_i**3-b_i**3)+x[1]*.5*(c_i**2-b_i**2))
                arg2 = cap*np.pi*(c_i**2-b_i**2)
                if (arg <= 0) or (arg2 <= 0):
                    obj += 10000
                else:
                    obj += housing_pref[index-1]*np.log(arg)+(1-housing_pref[index-1])*np.log(arg2)-U
            obj = abs(obj)
            return obj

        def demand_nonlinear(x):
            cons = []
            for index in gov_list:
                income = incomes[index-1]
                c_i = row['coords'][0][2*(index-1)+1]
                b_i = row['coords'][0][2*(index-1)]
                cap = capital(j-1,x[0],x[1],alpha[j-1],gamma[j-1],tau[j-1])**d
                cons.append(1-2*np.pi*(1/income)*cap*(x[0]*(1/3)*(c_i**3-b_i**3)+x[1]*.5*(c_i**2-b_i**2)))
            return np.asarray(cons)

        def optimize_prices(row):
            gov_lists = {'gov1':row['gov1'],'gov2':row['gov2']}
            bounds = Bounds([-np.inf, 0], [np.inf, np.inf])
            f = 0
            global j
            flag = 0
            for j in (1,2):
                global gov_list 
                gov_list = gov_lists[f'gov{j}']
                if len(gov_list)!=0:
                    linear_constraint = LinearConstraint([boundaries[j-1][1],1], 0, np.inf)
                    nonlinear_constraint = NonlinearConstraint(demand_nonlinear, 0, 1)
                    x0 = np.asarray([-1,100])
                    res = minimize(demand_objective,x0,bounds=bounds,constraints=[linear_constraint,nonlinear_constraint])
                    price = res.x
                    f += demand_objective(price)

                    cap = capital(j-1,price[0],price[1],alpha[j-1],gamma[j-1],tau[j-1])**d
                    last_ind = gov_list[-1]
                    c_i = row['coords'][0][2*(last_ind-1)+1]
                    b_i = row['coords'][0][2*(last_ind-1)]
                    thresh = 1.3*cap*np.pi*(c_i**2-b_i**2)-beta[j-1]
                    if thresh < 0:
                        flag = 1
            return f, flag

        success = False
        attempts = 0
        while not success and attempts<5:
            attempts+=1

            gov_payoff = {'gov1':0,'gov2':0,'boundary':[boundarycondition],'profits':'undefined','strat1':f'{alpha[0]},{beta[0]},{tau[0]},{gamma[0]}','strat2':f'{alpha[1]},{beta[1]},{tau[1]},{gamma[1]}','govexp1':'undefined','govexp2':'undefined','govpop1':'undefined','govpop2':'undefined','govinfluence1':'undefined','govinfluence2':'undefined'}

            data = options_df[options_df['in_city']==N]

            counter = 0
            minf = 10000000000000000
            minind = 0
            term_flag = 1
            while (term_flag==1):
                for index, row in data.iterrows():
                    f, flag = optimize_prices(row)
                    if f < minf:
                        minf = f
                        minind = row
                        term_flag = flag
                if term_flag==1:
                    counter+=1
                    if counter<N:
                        data = options_df[options_df['in_city']==N-counter]
                    else:
                        term_flag = 0
                        gov_payoff['gov1'] = 0 
                        gov_payoff['gov2'] = 0
                else:
                    term_flag = 0

            def objective(x):
                obj = 0

                p_j = x[-2]
                s_j = x[-1]

                gov_revenues=gamma[j-1]*((1/3)*p_j*(boundaries[j-1][1]**3-boundaries[j-1][0]**3)+.5*s_j*(boundaries[j-1][1]**2-boundaries[j-1][0]**2))
                pop_gov = len(gov_list)
                
                counter = 0
                for i in gov_list:
                    counter+=1
                    income = housing_pref[i-1]*np.log(incomes[i-1])
                    pg = pg_pref[i-1]*(2*np.pi)*(gov_revenues/pop_gov)
                    if counter ==1:
                        util = utility_pref[i-1]*(.5*R*(x[0]**2-boundaries[j-1][0]**2)-(1/3)*(x[0]**3-boundaries[j-1][0]**3))
                    elif counter==len(gov_list):
                        util = utility_pref[i-1]*(.5*R*(boundaries[j-1][1]**2-x[counter-2]**2)-(1/3)*(boundaries[j-1][1]**2-x[counter-2]**2))
                    else:
                        util = utility_pref[i-1]*(.5*R*(x[counter-1]**2-x[counter-2]**2)-(1/3)*(x[counter-1]**2-x[counter-2]**2))
                    
                    obj += income+U+util+pg

                obj *= -1
                return obj

            def beta_constraint(x):
                cons = []
                counter = 0
                for i in gov_list:
                    counter+=1
                    if counter ==1:
                        choice = (boundaries[j-1][0],x[0])
                    elif counter==len(gov_list):
                        choice = (x[counter-2],boundaries[j-1][1])
                    else:
                        choice = (x[counter-2],x[counter-1])
                    cap = capital(j-1,x[-2],x[-1],alpha[j-1],gamma[j-1],tau[j-1])**d
                    if cap ==0:
                        cons.append(0)
                    else:
                        cons.append(-1*choice[1]**2+choice[0]**2-(beta[j-1]*(1/np.pi)*(1/cap)))
                return np.asarray(cons)

            def demand_constraint(x):
                cons = []
                counter = 0
                for i in gov_list:
                    income = incomes[i-1]
                    counter+=1
                    if counter ==1:
                        c_i = x[0]
                        b_i = boundaries[j-1][0]
                    elif counter==len(gov_list):
                        c_i = boundaries[j-1][1]
                        b_i = x[counter-2]
                    else:
                        c_i = x[counter-1]
                        b_i = x[counter-2]
                    cap= capital(j-1,x[-2],x[-1],alpha[j-1],gamma[j-1],tau[j-1])**d
                    arg = 1-2*np.pi*(1/income)*cap*(x[-2]*(1/3)*(c_i**3-b_i**3)+x[-1]*.5*(c_i**2-b_i**2))
                    arg2 = cap*np.pi*(c_i**2-b_i**2)
                    if (arg <= 0) or (arg2 <= 0):
                        cons.append(0)
                    else:
                        cons.append(housing_pref[i-1]*np.log(arg)+(1-housing_pref[i-1])*np.log(arg2)-U)
                
                return np.asarray(cons)

            def price_constraint(x):
                cons = []
                counter = 0
                for i in gov_list:
                    income = incomes[i-1]
                    counter+=1
                    if counter ==1:
                        c_i = x[0]
                        b_i = boundaries[j-1][0]
                    elif counter==len(gov_list):
                        c_i = boundaries[j-1][1]
                        b_i = x[counter-2]
                    else:
                        c_i = x[counter-1]
                        b_i = x[counter-2]
                    cap = capital(j-1,x[-2],x[-1],alpha[j-1],gamma[j-1],tau[j-1])**d
                    cons.append(1-2*np.pi*(1/income)*cap*(x[-2]*(1/3)*(c_i**3-b_i**3)+x[-1]*.5*(c_i**2-b_i**2)))
                return np.asarray(cons)

            gov_lists = {'gov1':minind['gov1'],'gov2':minind['gov2']}

            res_list = {}
            obj_list = {}
            successes = 1
            for j in (1,2):
                gov_list = gov_lists[f'gov{j}']
                if len(gov_list)>1:
                    upper_bound = [boundaries[j-1][1] for x in range(len(gov_list)+1)]
                    upper_bound[-1] = np.inf
                    upper_bound[-2] = np.inf
                    lower_bound = [boundaries[j-1][0] for x in range(len(gov_list)+1)]
                    lower_bound[-1] = 0
                    lower_bound[-2] = -np.inf
                    bounds = Bounds(lower_bound,upper_bound)

                    padding = np.zeros(len(gov_list) - 1)
                    first_col = np.r_[[1,-1], padding]
                    linear1 = toeplitz(first_col)
                    linear1 = np.triu(linear1)
                    linear1[-1] = np.zeros(len(gov_list) + 1)
                    linear1[-2] = np.zeros(len(gov_list) + 1)
                    linear1 = LinearConstraint(linear1, -np.inf, 0)
                    
                    linear2 = np.zeros((len(gov_list)+1,len(gov_list)+1))
                    linear2[-2,-2] = boundaries[j-1][1]
                    linear2[-2,-1] = 1
                    linear2 = LinearConstraint(linear2, 0, np.inf)

                    beta_constraint = NonlinearConstraint(beta_constraint, -np.inf, 0)
                    demand_constraint = NonlinearConstraint(demand_constraint, -1, 1)
                    price_constraint = NonlinearConstraint(price_constraint, 0, 1)
                    
                    slice = (boundaries[j-1][1]-boundaries[j-1][0])/len(gov_list)
                    x0 = np.zeros((len(gov_list)-1))
                    counter = 0
                    for x in range(0,len(gov_list)-1):
                        x0[counter] = boundaries[j-1][0]+slice*(counter+1)
                        counter += 1
                    x0 = np.append(x0, np.asarray([-1,100]))
                    try:
                        res = minimize(objective,x0,bounds=bounds,constraints=[linear1,linear2,beta_constraint,demand_constraint,price_constraint])
                    except:
                        res.x = [0,0]
                    res_list[f'gov{j}'] = res.x
                    obj_list[f'gov{j}'] = objective(res.x)
                    if not res.success:
                        successes *= 0
                
                if len(gov_list)==1:
                    bounds = Bounds([-np.inf, 0], [np.inf, np.inf])
                    def demand_objective(x):
                        obj = 0
                        for index in gov_list:
                            income = incomes[index-1]
                            c_i = boundaries[j-1][1]
                            b_i = boundaries[j-1][0]
                            cap= capital(j-1,x[0],x[1],alpha[j-1],gamma[j-1],tau[j-1])**d
                            arg = 1-2*np.pi*(1/income)*cap*(x[0]*(1/3)*(c_i**3-b_i**3)+x[1]*.5*(c_i**2-b_i**2))
                            arg2 = cap*np.pi*(c_i**2-b_i**2)
                            if (arg <= 0) or (arg2 <= 0):
                                obj += 10000
                            else:
                                obj += housing_pref[index-1]*np.log(arg)+(1-housing_pref[index-1])*np.log(arg2)-U
                        obj = abs(obj)
                        return obj
                    
                    def demand_nonlinear(x):
                        cons = []
                        for index in gov_list:
                            income = incomes[index-1]
                            c_i = boundaries[j-1][1]
                            b_i = boundaries[j-1][0]
                            cap = capital(j-1,x[0],x[1],alpha[j-1],gamma[j-1],tau[j-1])**d
                            cons.append(1-2*np.pi*(1/income)*cap*(x[0]*(1/3)*(c_i**3-b_i**3)+x[1]*.5*(c_i**2-b_i**2)))
                        return np.asarray(cons)
                    
                    linear_constraint = LinearConstraint([boundaries[j-1][1],1], 0, np.inf)
                    nonlinear_constraint = NonlinearConstraint(demand_nonlinear, 0, 1)
                    x0 = np.asarray([-1,100])
                    res = minimize(demand_objective,x0,bounds=bounds,constraints=[linear_constraint,nonlinear_constraint])
                    res_list[f'gov{j}'] = res.x
                    obj_list[f'gov{j}'] = objective([boundaries[j-1][0],boundaries[j-1][1],res.x[0],res.x[1]])
                    if not res.success:
                        successes *= 0

                if len(gov_list)==0:
                    res_list[f'gov{j}'] = [0,0]
                    obj_list[f'gov{j}'] = 0

            if successes == 1:
                success = True
            elif attempts == 5 and success==False:
                gov_payoff['gov1'] = -10000
                gov_payoff['gov2'] = -10000
                continue
            else:
                continue
            
            incomes1 = 0
            for ind in gov_lists['gov1']:
                incomes1+=incomes[ind-1]
            incomes2 = 0
            for ind in gov_lists['gov2']:
                incomes2+=incomes[ind-1]
            
            def q(j,p_j,s_j,alpha_j,tau_j,gamma_j):
                cap = capital(j,p_j,s_j,alpha[j-1],gamma[j-1],tau[j-1])
                delta_2 = boundaries[j-1][1]**2 - boundaries[j-1][0]**2
                delta_3 = boundaries[j-1][1]**3 - boundaries[j-1][0]**3

                x = (1/3)*p_j*delta_3+(1/2)*s_j*delta_2
                if x <=0:
                    return (2/3)*(delta_3/delta_2)
                if 0 < x <= (1/2)*((alpha_j/(np.pi*delta_2))**((1-d)/d))*(1/d)*(1/(1-gamma_j))*delta_2*(k+tau_j):
                    return (2/delta_2)*((1-gamma_j)*((1/3)*p_j*delta_3+.5*s_j*delta_2)*(cap**d)-.5*delta_2*(k+tau_j)*cap+(delta_3/3))
                if (1/2)*((alpha_j/(np.pi*delta_2))**((1-d)/d))*(1/d)*(1/(1-gamma_j))*delta_2*(k+tau_j)<x:
                    return (cap**d)*(k+tau_j)*(1/d)*((alpha_j/(np.pi*delta_2))**((1-d)/d))-(k+tau_j)*cap+(2/3)*(delta_3/delta_2)

            if incomes1>0 and res_list['gov1'][-2]!=0:
                p_1 = res_list['gov1'][-2]
                s_1 = res_list['gov1'][-1]

                cap1 = capital(1,p_1,s_1,alpha[0],gamma[0],tau[0])
                delta_2_1 = boundaries[0][1]**2 - boundaries[0][0]**2
                delta_3_1 = boundaries[0][1]**3 - boundaries[0][0]**3

                q_1 = q(1,p_1,s_1,alpha[0],tau[0],gamma[0])

                gov_payoff['govexp1'] = gamma[0]*((p_1/3)*delta_3_1+(s_1/2)*delta_2_1)


            if incomes2>0 and res_list['gov2'][-2]!=0:
                p_2 = res_list['gov2'][-2]
                s_2 = res_list['gov2'][-1]

                cap2 = capital(2,p_2,s_2,alpha[1],gamma[1],tau[1])
                delta_2_2 = boundaries[1][1]**2 - boundaries[1][0]**2
                delta_3_2 = boundaries[1][1]**3 - boundaries[1][0]**3
                
                q_2 = q(2,p_2,s_2,alpha[1],tau[1],gamma[1])

                gov_payoff['govexp2'] = gamma[1]*((p_2/3)*delta_3_2+(s_2/2)*delta_2_2)

            profits = 0
            if incomes1>0 and res_list['gov1'][-2]!=0:
                profits += (1-gamma[0])*((1/3)*p_1*delta_3_1+.5*s_1*delta_2_1)*cap1**d-.5*delta_2_1*(k+tau[0])*cap1+delta_3_1/3-(delta_2_1/2)*q_1
            if incomes2>0 and res_list['gov2'][-2]!=0:
                profits += (1-gamma[1])*((1/3)*p_2*delta_3_2+.5*s_2*delta_2_2)*cap2**d-.5*delta_2_2*(k+tau[1])*cap2+delta_3_2/3-(delta_2_2/2)*q_2
            profits *= 2*np.pi
            
            gov_payoff['profits'] = profits
            gov_payoff['govpop1'] = len(gov_lists['gov1'])
            gov_payoff['govpop2'] = len(gov_lists['gov2'])
            
            influence1 = math.exp(-math.exp(.1-.1*incomes1/(profits+1)))
            influence2 = math.exp(-math.exp(.1-.1*incomes2/(profits+1)))

            gov_payoff['govinfluence1'] = influence1
            gov_payoff['govinfluence2'] = influence2

            if incomes1 > 0 and res_list['gov1'][-2]!=0:
                gov_payoff['gov1'] = -influence1*obj_list['gov1']+(1-influence1)*(profits+1)
            if incomes2 >0 and res_list['gov2'][-2]!=0:
                gov_payoff['gov2'] = -influence2*obj_list['gov2']+(1-influence2)*(profits+1)

        governmental_analysis = pd.concat([governmental_analysis,pd.DataFrame.from_dict(gov_payoff)])
        governmental_analysis.to_csv('governmental_analysis.csv',index=False)