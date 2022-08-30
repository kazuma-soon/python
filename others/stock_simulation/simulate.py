import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import basic_plan
import percent_plan
import down_plan

# ハイパーパラメータ
span   = 20
budget = 900000
expected_return = -0.05


gbm_n  = (span * 12) + 1
plan_n = gbm_n - 1

data_y = utils.gbm(expected_return=expected_return, span=span, n=gbm_n)
basic_plan_invest,   basic_plan_principal   = basic_plan.basic_plan(budget=budget, n=plan_n, data_y=data_y)
percent_plan_invest, percent_plan_principal = percent_plan.percent_plan(budget=budget, n=plan_n, data_y=data_y)
down_plan_invest,    down_plan_principal    = down_plan.down_plan(budget=budget, n=plan_n, data_y=data_y)


fig, ax = plt.subplots()

ax.set_xlabel('month')
ax.set_ylabel('y')
ax.set_xlim([0, plan_n])

month_x = np.linspace(0, plan_n, plan_n)
# ax.plot(month_x, price,                color='blue',   label='price')
ax.plot(month_x, basic_plan_invest,    color='orange', label='basic-invest')
ax.plot(month_x, basic_plan_principal, color='orange', label='basic-principal', linestyle='dotted')

ax.plot(month_x, percent_plan_invest,    color='green', label='percent-invest')
ax.plot(month_x, percent_plan_principal, color='green', label='percent-principal', linestyle='dotted')

ax.plot(month_x, down_plan_invest,    color='blue', label='down_invest')
ax.plot(month_x, down_plan_principal, color='blue', label='down-principal', linestyle='dotted')

ax.legend(loc=0)
plt.savefig("others/stock_simulation/simulate.png")
plt.show()



def create_price_graph():
    x = np.linspace(1, 242, 242)
    price_y = utils.gbm(expected_return=expected_return, span=20.0, n=242)
    plt.plot(x, price_y, color='red', label='price')
    plt.legend(loc=0)
    plt.savefig("others/stock_simulation/price.png")
    plt.show()
    

create_price_graph()
