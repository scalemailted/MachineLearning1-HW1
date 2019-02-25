
/*
A = np.matrix([[4.0, 2.0], [2.0, 2.0]])
b = np.matrix([[-1.0], [1.0]])  # we will use the convention that a vector is a column vector
c = 0.0
*/


var A = nj.array([[4.0, 2.0], [2.0, 2.0]]);
var b = nj.array([-1.0,1.0]);  // we will use the convention that a vector is a column vector
var c = 0.0;

function f(x, A, b, c){
    //return float(0.5 * x.T * A * x - b.T * x + c)
    let xAx = nj.dot( nj.dot(x.T,A), x);
    let bx = nj.dot(b.T,x)
    return 0.5 * (xAx.get(0) - bx.get(0) + c)
}

/*function bowl(A, b, c){
    fig = plt.figure(figsize=(10,8))
    qf = fig.gca(projection='3d')
    size = 20
    x1 = list(np.linspace(-6, 6, size))
    x2 = list(np.linspace(-6, 6, size))
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, A, b, c)
    qf.plot_surface(x1, x2, zs, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0)
    fig.show()
    return x1, x2, zs
}*/

/*
Plotly.d3.csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv', function(err, rows){
function unpack(rows, key) {
  return rows.map(function(row) { return row[key]; });
}
  
var z_data=[ ]
for(i=0;i<24;i++)
{
  z_data.push(unpack(rows,i));
}
*/

function meshgrid(x,y)
{
	let rows = y.length;
	let cols = x.length;
	let xgrid = [];
	for (let i=0; i< rows; i++)
	{
		xgrid.push(x);
	}
	let ygrid = [];
	for (let j=0; j< rows; j++)
	{
		let row = []
		for (let c=0; c<cols; c++)
		{
			row.push(y[j])
		}
		ygrid.push(row)
	}
	return [ygrid, xgrid];
}

function bowl(A,b,c)
{
	let size = 20;
	//let x1 = list(np.linspace(-6, 6, size))
	//let x2 = list(np.linspace(-6, 6, size))
	//let x1 = [-6.0, -5.368421052631579, -4.7368421052631575, -4.105263157894737, -3.473684210526316, -2.8421052631578947, -2.210526315789474, -1.578947368421053, -0.9473684210526319, -0.3157894736842106, 0.3157894736842106, 0.947368421052631, 1.5789473684210522, 2.2105263157894726, 2.842105263157894, 3.473684210526315, 4.105263157894736, 4.7368421052631575, 5.368421052631579, 6.0] 
	//let x2 = [-6.0, -5.368421052631579, -4.7368421052631575, -4.105263157894737, -3.473684210526316, -2.8421052631578947, -2.210526315789474, -1.578947368421053, -0.9473684210526319, -0.3157894736842106, 0.3157894736842106, 0.947368421052631, 1.5789473684210522, 2.2105263157894726, 2.842105263157894, 3.473684210526315, 4.105263157894736, 4.7368421052631575, 5.368421052631579, 6.0] 
	let x1 = [-2.0, -1.7894736842105263, -1.5789473684210527, -1.368421052631579, -1.1578947368421053, -0.9473684210526316, -0.736842105263158, -0.5263157894736843, -0.3157894736842106, -0.10526315789473695, 0.10526315789473673, 0.3157894736842106, 0.5263157894736841, 0.7368421052631575, 0.9473684210526314, 1.1578947368421053, 1.3684210526315788, 1.5789473684210522, 1.789473684210526, 2.0]
	let x2 = [-2.0, -1.7894736842105263, -1.5789473684210527, -1.368421052631579, -1.1578947368421053, -0.9473684210526316, -0.736842105263158, -0.5263157894736843, -0.3157894736842106, -0.10526315789473695, 0.10526315789473673, 0.3157894736842106, 0.5263157894736841, 0.7368421052631575, 0.9473684210526314, 1.1578947368421053, 1.3684210526315788, 1.5789473684210522, 1.789473684210526, 2.0]
	let z = nj.zeros([size, size]);
	for (let i in [...Array(size).keys()] )
	{
        for (let j in [...Array(size).keys()] )
        {
            //let x = nj.array([[x1[i,j]], [x2[i,j]]])
            //console.log("x1: " + x1[i] + ", x2: " + x2[j]);
            let x = nj.array([x1[i],x2[j]]);
            let val = f(x, A, b, c);
            //console.log(val)
            z.set(i,j, val);
        }
    }
    
    return z.tolist();
}

//let x1 = [-6.0, -5.368421052631579, -4.7368421052631575, -4.105263157894737, -3.473684210526316, -2.8421052631578947, -2.210526315789474, -1.578947368421053, -0.9473684210526319, -0.3157894736842106, 0.3157894736842106, 0.947368421052631, 1.5789473684210522, 2.2105263157894726, 2.842105263157894, 3.473684210526315, 4.105263157894736, 4.7368421052631575, 5.368421052631579, 6.0]
//let x2 = [-6.0, -5.368421052631579, -4.7368421052631575, -4.105263157894737, -3.473684210526316, -2.8421052631578947, -2.210526315789474, -1.578947368421053, -0.9473684210526319, -0.3157894736842106, 0.3157894736842106, 0.947368421052631, 1.5789473684210522, 2.2105263157894726, 2.842105263157894, 3.473684210526315, 4.105263157894736, 4.7368421052631575, 5.368421052631579, 6.0] 
let x1 = [-2.0, -1.7894736842105263, -1.5789473684210527, -1.368421052631579, -1.1578947368421053, -0.9473684210526316, -0.736842105263158, -0.5263157894736843, -0.3157894736842106, -0.10526315789473695, 0.10526315789473673, 0.3157894736842106, 0.5263157894736841, 0.7368421052631575, 0.9473684210526314, 1.1578947368421053, 1.3684210526315788, 1.5789473684210522, 1.789473684210526, 2.0]
let x2 = [-2.0, -1.7894736842105263, -1.5789473684210527, -1.368421052631579, -1.1578947368421053, -0.9473684210526316, -0.736842105263158, -0.5263157894736843, -0.3157894736842106, -0.10526315789473695, 0.10526315789473673, 0.3157894736842106, 0.5263157894736841, 0.7368421052631575, 0.9473684210526314, 1.1578947368421053, 1.3684210526315788, 1.5789473684210522, 1.789473684210526, 2.0]

var grid_data = meshgrid(x1, x2);
var x_data = grid_data[0]
var y_data = grid_data[1] 
var z_data= bowl(A,b,c);





var data = [{
			x: x_data,
			y: y_data,
           	z: z_data,
           	type: 'surface',
           	opacity:0.9,
           	colorbar: {len:0.5, thickness:10, xpad:30 }
        }];
  
var layout = {
  title: 'Steepest Descent vs Conjugate Gradient',//'Conjugate Gradient',
  scene: {
		xaxis:{title: 'x1'},
		yaxis:{title: 'x2'},
		zaxis:{title: 'y'},
		},
  autosize: false,
  width: 500,
  height: 500,
  margin: {
    l: 65,
    r: 50,
    b: 65,
    t: 90,
  }
};


Plotly.newPlot('3d-plot', data, layout);


/*3d line for descent steps*/

var xline = []//[0,6]
var yline = []//[0,6]
var zline = []//[0,100]

var steps = steepestDescent();
for (point of steps){
	let x_coord = point[0]
	let y_coord = point[1]
	let x1x2 = nj.array([x_coord,y_coord])
	let z_coord = f(x1x2, A, b, c);
	console.log("x: " + x_coord + ", y: " + y_coord + "z: " + z_coord)
	xline.push(x_coord)
	yline.push(y_coord)
	zline.push(z_coord)
}

Plotly.plot('3d-plot', [{
  name: 'Steepest Descent',
  type: 'scatter3d',
  mode: 'lines',
  x: xline,
  y: yline,
  z: zline,
  opacity: 1,
  line: {
    width: 6,
    color: 'lightgreen',
    reversescale: false
  }
}], {
  height: 640
});


/*Steepest Descent*/
function steepestDescent()
{
	let x = nj.array([-2.0,-2.0])
	let steps = [[-2.0, -2.0]]
	let i = 0
	let imax = 6
	let eps = 0.01
	let r = nj.subtract(b, nj.dot(A,x) ) 
	let delta = nj.dot(r.T,r)
	let delta0 = delta.get(0)
	while (i < imax )//&& delta.get(0) > eps**2 * delta0)
	{
		let rAr =  nj.dot(r.T, nj.dot(A,r) );
	    let alpha = delta.get(0) / rAr.get(0) ;
	    x = nj.add(x, nj.multiply(r,alpha))
	    steps.push( [x.get(0), x.get(1)] )  //# store steps for future drawing
	    r = nj.subtract(b, nj.dot(A,x) );
	    delta = nj.dot(r.T,r)
	    i += 1
	}
	return steps

}











/*conjugate Gradient*/
function conjugateGradient()
{
	let x = nj.array([-2.0,-2.0])			//x = np.matrix([[-2.0],[-2.0]])
	let steps = [[-2.0, -2.0]]				//steps = [(-2.0, -2.0)]
	let i = 0     							//i = 0
	let imax = 2 							//imax = 10
	let eps = 0.01 							//eps = 0.01
	let r = nj.subtract(b, nj.dot(A,x) ) 	//r = b - A * x
	let d = r 								//d = r
	let deltanew = nj.dot(r.T,r) 			//deltanew = r.T * r
	let delta0 = deltanew.get(0) 				//delta0 = deltanew
	while (i < imax )//&& delta.get(0) > eps**2 * delta0)
	{
		let dAd = nj.dot( d.T, nj.dot(A,d) );	//let rAr =  nj.dot(r.T, nj.dot(A,r) );
		alpha = deltanew.get(0) / dAd.get(0);	//let alpha = delta.get(0) / rAr.get(0) ;
   		x = nj.add(x, nj.multiply(d, alpha)); 	//x = nj.add(x, nj.multiply(r,alpha))
	    steps.push( [x.get(0), x.get(1)] )  	//# store steps for future drawing
	    r = nj.subtract(b, nj.dot(A,x) ); 		//r = b - A * x
	    deltaold = deltanew
	    deltanew = nj.dot(r.T,r)				//delta = nj.dot(r.T,r)
	    beta = deltanew.get(0) / deltaold.get(0)
	    d = nj.add(r, nj.multiply(d, beta))
	    i += 1
	}
	return steps

}


/*3d line for descent steps*/

xline = []
yline = []
zline = []

steps = conjugateGradient();
for (point of steps){
	let x_coord = point[0]
	let y_coord = point[1]
	let x1x2 = nj.array([x_coord,y_coord])
	let z_coord = f(x1x2, A, b, c);
	console.log("x: " + x_coord + ", y: " + y_coord + "z: " + z_coord)
	xline.push(x_coord)
	yline.push(y_coord)
	zline.push(z_coord)
}

Plotly.plot('3d-plot', [{
  name: 'Conjugate Gradient',
  type: 'scatter3d',
  mode: 'lines',
  x: xline,
  y: yline,
  z: zline,
  opacity: 1,
  line: {
    width: 6,
    color: 'orange',
    reversescale: false
  }
}], {
  height: 640
});





/*
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
b = np.matrix([[2.0], [-8.0]])  # we will use the convention that a vector is a column vector
c = 0.0

def f(x, A, b, c):
    return float(0.5 * x.T * A * x - b.T * x + c)

def bowl(A, b, c):
    fig = plt.figure(figsize=(10,8))
    qf = fig.gca(projection='3d')
    size = 20
    x1 = list(np.linspace(-6, 6, size))
    x2 = list(np.linspace(-6, 6, size))
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, A, b, c)
    qf.plot_surface(x1, x2, zs, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0)
    fig.show()
    return x1, x2, zs

x1, x2, zs = bowl(A, b, c);

def contoursteps(x1, x2, zs, steps=None):
    fig = plt.figure(figsize=(6,6))
    cp = plt.contour(x1, x2, zs, 10)
    plt.clabel(cp, inline=1, fontsize=10)
    if steps is not None:
        steps = np.matrix(steps)
        plt.plot(steps[:,0], steps[:,1], '-o')
    fig.show();

contoursteps(x1, x2, zs);


x = np.matrix([[-2.0],[-2.0]])
steps = [(-2.0, -2.0)]
i = 0
imax = 10
eps = 0.01
r = b - A * x
delta = r.T * r
delta0 = delta
while i < imax and delta > eps**2 * delta0:
    alpha = float(delta / (r.T * (A * r)))
    x = x + alpha * r
    steps.append((x[0,0], x[1,0]))  # store steps for future drawing
    r = b - A * x
    delta = r.T * r
    i += 1

contoursteps(x1, x2, zs, steps)

x = np.matrix([[-2.0],[-2.0]])
steps = [(-2.0, -2.0)]
i = 0
imax = 10
eps = 0.01

print 'b:\n', b
print 'A:\n', A
print 'x:\n', x
r = b - A * x
print 'r:\n', r
contoursteps(x1, x2, zs, None)
plt.plot([0, r[0, 0] * 0.5], [0, r[1, 0] * 0.5], 'g')
plt.show()

print '||r||^2 =', np.linalg.norm(r)**2
delta = r.T * r
print 'r.T * r = ',  delta
delta0 = delta

alpha = float(delta / (r.T * A * r))

x = x + alpha * r
contoursteps(x1, x2, zs, [(-2, -2), (x[0, 0], x[1, 0])])
plt.plot([0, r[0, 0] * 0.5], [0, r[1, 0] * 0.5], 'g')
plt.show()

r = b - A * x
contoursteps(x1, x2, zs, [(-2, -2), (x[0, 0], x[1, 0])])
plt.plot([0, r[0, 0]], [0, r[1, 0]], 'g')


delta = r.T * r
i += 1


x = np.matrix([[-2.0],[-2.0]])
steps = [(-2.0, -2.0)]
i = 0
imax = 10000
eps = 0.01
alpha = 0.12  # play with this value to see how it affects the optimization process, try 0.05, 0.27 and 0.3
r = b - A * x
delta = r.T * r
delta0 = delta
while i < imax and delta > eps**2 * delta0:
    x = x + alpha * r
    steps.append((x[0,0], x[1,0]))  # store steps for future drawing
    r = b - A * x
    delta = r.T * r
    i += 1

contoursteps(x1, x2, zs, steps)

Around = np.matrix([[1, 0],[0, 1]])
bround = np.matrix([[0],[0]])
cround = 0
x1, x2, zs = bowl(Around, bround, cround)

va = np.matrix([[2],[2]])
vb = np.matrix([[2],[-2]])
contoursteps(x1, x2, zs, [(va[0,0],va[1,0]),(0,0),(vb[0,0],vb[1,0])])

Ascaled = np.matrix([[1, 0],[0, 2]])
bscaled = np.matrix([[0],[0]])
cscaled = 0
x1, x2, zs = bowl(Ascaled, bscaled, cscaled)

va = np.matrix([[2],[np.sqrt(2)]])
vb = np.matrix([[2],[-np.sqrt(2)]])
contoursteps(x1, x2, zs, [(va[0,0],va[1,0]),(0,0),(vb[0,0],vb[1,0])])

Around = np.matrix([[1, 0],[0, 1]])
bround = np.matrix([[0],[0]])
cround = 0
x1, x2, zs = bowl(Around, bround, cround)


x = np.matrix([[-2.0],[-2.0]])
steps = [(-2.0, -2.0)]
i = 0
imax = 10
eps = 0.01
r = bround - np.matrix([[1, 0],[0, 0]]) * x  # replaced Around with this to force residual to be parallel to X axis
delta = r.T * r
delta0 = delta
while i < imax and delta > eps**2 * delta0:
    alpha = float(delta / (r.T * (Around * r)))
    x = x + alpha * r
    steps.append((x[0,0], x[1,0]))  # store steps for future drawing
    r = bround - Around * x
    delta = r.T * r
    i += 1

contoursteps(x1, x2, zs, steps)

x1, x2, zs = bowl(A, b, c)

x = np.matrix([[-2.0],[-2.0]])
steps = [(-2.0, -2.0)]
i = 0
imax = 10
eps = 0.01
r = b - A * x
d = r
deltanew = r.T * r
delta0 = deltanew
while i < imax and deltanew > eps**2 * delta0:
    alpha = float(deltanew / float(d.T * (A * d)))
    x = x + alpha * d
    steps.append((x[0, 0], x[1, 0]))
    r = b - A * x
    deltaold = deltanew
    deltanew = r.T * r
    #beta = -float((r.T * A * d) / float(d.T * A * d))
    beta = float(deltanew / float(deltaold))
    d = r + beta * d
    i += 1


contoursteps(x1, x2, zs, steps)

x = np.matrix([[-2.0],[-2.0]])
steps = [(-2.0, -2.0)]
i = 0
imax = 10
eps = 0.01
r = b - A * x
d = r
deltanew = r.T * r
delta0 = deltanew
#while i < imax and deltanew > eps**2 * delta0:


alpha = float(deltanew / float(d.T * (A * d)))
x = x + alpha * d
steps.append((x[0, 0], x[1, 0]))


r = b - A * x

contoursteps(x1, x2, zs, [(-2, -2), (x[0,0],x[1,0])])
plt.plot([0, r[0, 0]], [0, r[1, 0]])
plt.show()

d.T * A * r

deltaold = deltanew
deltanew = r.T * r
beta = float(deltanew / float(deltaold))

oldd = d  # this is needed for future demonstration, not relevant to the algorithm
d = r + beta * d

contoursteps(x1, x2, zs, [(-2, -2), (x[0,0],x[1,0])])
plt.plot([0,d[0,0]], [0, d[1,0]])  # new direction
plt.plot([0,r[0,0]], [0, r[1,0]])  # old direction (residual)
plt.show()

oldd.T * A * d

i += 1

alpha = float(deltanew / float(d.T * (A * d)))
x = x + alpha * d
steps.append((x[0, 0], x[1, 0]))
contoursteps(x1, x2, zs, steps)
plt.plot([0,d[0,0]], [0, d[1,0]])
plt.show()
*/

