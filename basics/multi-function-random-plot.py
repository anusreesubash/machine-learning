import matplotlib.pyplot as plt
import numpy as np

data = {
	'x' : np.arange(50),
	'y' :np.random.randint(0, 50, 50),
	'colors' : np.random.randint(0, 50, 50),
	'size' : np.random.randn(50)
}

data['size'] = np.abs(data['size']) * 100

plt.scatter('x', 'y', c='colors', s='size',data=data)
plt.xlabel('entry x')
plt.ylabel('entry y')
plt.show()