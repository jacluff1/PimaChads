import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb

class Downpayment:

    def __init__(self, cost):
        self.cost = cost
        self.minimum = cost * 0.2

    def __check(self, x):
        assert np.all([ (x >= 0) , (x <= .8) ]), "please insert a value 0 <= x <= .8"

    def f(self, x):
        self.__check(x)
        return self.minimum + (x * self.cost)

    def dfdx(self, x):
        self.__check(x)
        return self.cost

class Loan:

    def __init__(self, sale_price, APR, years):

        # calculate some dependent variables
        closing_cost = sale_price * .03
        cost = sale_price + closing_cost

        # assign important variables as attributes
        self.cost = cost # the total initial cost of the house
        self.MPR = APR / 12 # the monthly percentage rate
        self.nper = years * 12 # number of periods in the loan
        self.downpayment = Downpayment(cost) # the downpayment

    def __check(self, x):
        assert np.all([ (x >= 0) , (x <= .8) ]), "please insert a value 0 <= x <= .8"

    def f(self, x):
        self.__check(x)
        return (self.cost - self.downpayment.f(x)) * (1 + self.MPR)**self.nper

    def dfdx(self, x):
        self.__check(x)
        return -self.downpayment.dfdx(x) * (1 + self.MPR)**self.nper

class RoE:

    def __init__(self, predicted, sale_price, APR, years):

        # calculate some dependent variables
        closing_cost = sale_price * .03
        cost = sale_price + closing_cost

        # assign important variables as attributes
        self.years = years
        self.closing_cost = closing_cost
        self.rental_rate = 0.1 * predicted
        self.predicted = predicted
        self.downpayment = Downpayment(cost)
        self.loan = Loan(sale_price, APR, years)

    def __check(self, x, t):
        assert t >= self.years*12, "we have made the basic assumption that the number of months will be greater than or equal to the lifetime of the loan"
        assert np.all([ (x >= 0) , (x <= .8) ]), "please insert a value 0 <= x <= .8"

    def profit(self, x, t):
        self.__check(x,t)
        return self.rental_rate*t + self.predicted - self.closing_cost - self.downpayment.f(x) - self.loan.f(x)

    def equity(self, x):
        return self.predicted - self.loan.f(x)

    def f(self,x,t):
        self.__check(x,t)
        return self.profit(x,t) / self.equity(x)

    def dfdx(self,x,t):

        self.__check(x,t)

        u = self.profit(x,t)
        dudx = -(self.downpayment.dfdx(x) + self.loan.dfdx(x))
        v = self.equity(x)
        dvdx = -self.loan.dfdx(x)

        return (v*dudx - u*dvdx) / v**2

class BusinessModel:

    def __init__(self):
        return

    def get_data(self):
        data = pd.read_csv("data/y_hat.csv")
        data['difference'] = data['predicted'] - data['saleprice']
        data.sort_values('difference', ascending=True, inplace=True)
        return data

def plot(t, APR=.05, years=30):

    predicted = np.arange(50,250,50) * 1000
    alpha = np.arange(.7,1.1,.1)

    nrows = predicted.shape[0]
    ncols = alpha.shape[0]
    N = int(predicted.shape[0] * alpha.shape[0])

    fig,ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    fig.suptitle(f"Profit after {t} years")

    for n in range(N):

        i,j = divmod(n,ncols)

        sale_price = alpha[j] * predicted[i]
        roe = RoE(predicted[i], sale_price, APR, years)
        print(predicted[i], sale_price, i, j)

        x = np.linspace(0,.8,1001)
        y = roe.f(x,t*12)
        idx = y.argmax()

        # ax[i,j].set_title("x: {:0.4f}: RoE: ${:0.2f}K".format(x[idx],y[idx]/1000))
        ax[i,j].plot(x,y)
        ax[i,j].set_xlim([.2,.8])
        ax[i,j].set_ylim(-y[idx])
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_title("x: {:0.4f}, RoE: ${:0.2f}K".format(x[idx],y[idx]/1000), fontsize=5)

    # plt.tight_layout()
    fig.subplots_adjust(hspace=.2, wspace=.2)
    fig.savefig(f"business_model/RoE.pdf")
    plt.close(fig)
