# 株価の上下に応じて買う額を変更するプラン

# 先月のmonthly_interestが99%なら10%多く買う

import numpy  as np
import pandas as pd
import utils


def percent_plan(init_invest=50000, continuous_investment=4000, budget=480000, n=120, data_y=utils.gbm()):
    investment  = init_invest
    principal   = init_invest
    investment_data_for_plot = np.array([])
    principal_data_for_plot  = np.array([])

    for t in range(n):
        monthly_interest = (data_y[t+1] - data_y[t]) / data_y[t] + 1
        investment *= monthly_interest

        if monthly_interest >= 1:
            add_investment = continuous_investment * (1 - (monthly_interest - 1)) * 2
            if budget >= add_investment:
                investment += add_investment
                principal  += add_investment
                budget -= add_investment
            else:
                investment += budget
                principal  += budget
                budget = 0

        if monthly_interest < 1:
            add_investment = continuous_investment * (1 + (1 - monthly_interest)) * 2
            if budget >= add_investment:
                investment += add_investment
                principal  += add_investment
                budget -= add_investment
            else:
                investment += budget
                principal  += budget
                budget = 0

        investment_data_for_plot, principal_data_for_plot = create_basic_plan_data_for_plot(investment_data_for_plot, principal_data_for_plot, investment, principal)

    investment = round(investment, 2)
    principal  = round(principal, 2)
    profits    = round(investment / principal, 2)
    print('---percent_plan---')
    print(f'{investment = }')
    print(f'{principal  = }')
    print(f'return     = {profits}')

    return investment_data_for_plot, principal_data_for_plot


def create_basic_plan_data_for_plot(investment_data_for_plot, principal_data_for_plot, investment, principal):
    investment_data_for_plot = np.append(investment_data_for_plot, investment)
    principal_data_for_plot  = np.append(principal_data_for_plot,  principal)
    return investment_data_for_plot, principal_data_for_plot

