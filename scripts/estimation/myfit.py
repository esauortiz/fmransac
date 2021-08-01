import math
import numpy as np

from numpy.linalg import inv, pinv
from scipy import optimize
from mpmath import acot
from matplotlib import pyplot as plt


def is_inlier(data, model_class, model_params, residual_threshold):
    data_model = model_class()
    data_residuals = np.abs(data_model.residuals(data, model_params))
    return data_residuals < residual_threshold

def get_residuals(data, model_class, model_params):
    data_model = model_class()
    data_residuals = np.abs(data_model.residuals(data, model_params))
    return data_residuals   

def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)

def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError(f'Input data must be at least 2D. Data shape is {data.shape}')

def _norm_along_axis(x, axis):
    """NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`."""
    return np.sqrt(np.einsum('ij,ij->i', x, x))

def musigma_norm(data):
    """Normalizes data based on
    mu-sigma normalization
    Parameters
    ----------
    data : (N, dim) array
        N points in a space of dimensionality dim >= 2.
    Returns
    -------
    data : (N, dim) array
        Normalized data
    H : (dim + 1, dim + 1) array
        Normalization matrix
    """
    dim = data.shape[1]

    # vector of means of every column in data
    mu_ax = data.mean(axis = 0)
    # vector of std of every column in data
    std_ax = data.std(axis = 0)

    # building normalization matrix H
    diag = np.ones((dim + 1,))
    diag[0:dim,] = 1 / std_ax
    H = np.diag(diag)
    H[:, dim][0:dim,] = -mu_ax/std_ax
    
    """ example of H matrix of 2D data
    
    H = np.array([  [1/s_x, 0,      -mu_x/s_x],
                    [0,     1/s_y,  -mu_y/s_y],
                    [0,     0,      1]])
    """

    # apply normalization and return
    data_matrix = np.ones((data.shape[0],data.shape[1] + 1))
    data_matrix[:,:-1] = data
    data =  np.dot(H, data_matrix.T).T[:,0:dim]
    return data, H

def maxmin_norm(data):
    """Normalizes data based on
    mu-sigma normalization
    Parameters
    ----------
    data : (N, dim) array
        N points in a space of dimensionality dim >= 2.
    Returns
    -------
    data : (N, dim) array
        Normalized data
    H : (dim + 1, dim + 1) array
        Normalization matrix
    """
    dim = data.shape[1]

    # vector of max of every column in data
    max_ax = data.max(axis = 0)
    # vector of min of every column in data
    min_ax = data.min(axis = 0)
    # vector of r of every column in data
    r_ax = max_ax - min_ax

    # building normalization matrix H
    diag = np.ones((dim + 1,))
    diag[0:dim,] = 1 / r_ax
    H = np.diag(diag)
    H[:, dim][0:dim,] = -min_ax/r_ax
    
    """ example of H matrix of 2D data
    
    H = np.array([  [1/r_x, 0,      -min_x/r_x],
                    [0,     1/r_y,  -min_y/r_y],
                    [0,     0,      1]])
    """

    # apply normalization and return
    data_matrix = np.ones((data.shape[0],data.shape[1] + 1))
    data_matrix[:,:-1] = data
    data =  np.dot(H, data_matrix.T).T[:,0:dim]
    return data, H

class BaseModel(object):

    def __init__(self):
        self.params = None

class LineModelND(BaseModel):
    """Total least squares estimator for N-dimensional lines.
    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.
    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::
        X = origin + lambda * direction
    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.
    """

    def estimate(self, data):
        """Estimate line model from data.
        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.
        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)

        origin = data.mean(axis=0)
        data = data - origin

        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            norm = np.linalg.norm(direction)
            if norm != 0:  # this should not happen to be norm 0
                direction /= norm
        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _,_, v = np.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        self.params = (origin, direction)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.
        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.
        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).
        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        _check_data_atleast_2D(data)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params
        norm = np.linalg.norm(direction)
        direction /= norm
        
        res = (data - origin) - \
              ((data - origin) @ direction)[..., np.newaxis] * direction
        return _norm_along_axis(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.
        Parameters
        ----------ip
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).
        Returns
        -------
        data : (n, m) array
            Predicted coordinates.
        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')
        
        origin, direction = params
        norm = np.linalg.norm(direction)
        direction /= norm

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError('Line parallel to axis %s' % axis)

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data

    def predict_x(self, y, params=None):
        """Predict x-coordinates for 2D lines using the estimated model.
        Alias for::
            predict(y, axis=1)[:, 0]
        Parameters
        ----------
        y : array
            y-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).
        Returns
        -------
        x : array
            Predicted x-coordinates.
        """
        x = self.predict(y, axis=1, params=params)[:, 0]
        return x

    def predict_y(self, x, params=None):
        """Predict y-coordinates for 2D lines using the estimated model.
        Alias for::
            predict(x, axis=0)[:, 1]
        Parameters
        ----------
        x : array
            x-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).
        Returns
        -------
        y : array
            Predicted y-coordinates.
        """
        y = self.predict(x, axis=0, params=params)[:, 1]
        return y

class PlaneModelND(BaseModel):
    """Total least squares estimator for N-dimensional lines.
    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.
    Planes are defined by a point (origin) and a unit vector (normal vector)
    according to the following vector equation::
        w'x + w_o = 0
    where w' is np.transpose(normal_vector) and w_o = w' * x_origin
    and x_origin is a point contained in the plane
    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `subspace`.
        where subspace constains the normal_vector and the rest of the axis of the
        subspace
    """

    def estimate(self, data, w = None):
        """Estimate line model from data.
        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.
        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """        
        _check_data_atleast_2D(data)
        """
        #TLS
        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            normal_vector = direction
            normal_vector[1] = -normal_vector[1] # perpendicular counter clockwise

            # as unitary vectors
            norm_direction = np.linalg.norm(direction)
            norm_normal_vector = np.linalg.norm(normal_vector)
            if (norm_direction != 0 and norm_normal_vector != 0):  # this should not happen to be norm 0
                normal_vector /= norm_normal_vector
                direction /= norm_direction

        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _,_, v = np.linalg.svd(data, full_matrices=False)
            normal_vector = v[(np.shape(v)[1]-1)]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        self.params = (origin, normal_vector)
        """

        #OWLS (if w = np.ones((points_num,)) results will be the same as TLS)
        if w is None:
            w = np.ones(np.shape(data[:,0]), dtype = int)
            w = w / np.sum(w)
        elif w.shape[0] != data.shape[0]:
            raise ValueError(f'Weigths array shape does not match data shape. w.shape[0] = {w.shape[0]} | data.shape[0] = {data.shape[0]}')
        """
        # particular solution 2D hiperplane: 2D line
        x = data[:,0]
        y = data[:,1]
        # Weighted orthogonal least squares fit of line a*x+b*y+c=0 to a set of 2D points with coordiantes given by x and y and weights w
        n = np.sum(w)
        meanx = np.sum(w*x)/n
        meany = np.sum(w*y)/n
        x = x - meanx
        y = y - meany

        y2x2 = np.sum(w*(y ** 2 - x ** 2))
        xy = np.sum(w*x*y)
        alpha = 0.5 * acot(0.5 * y2x2 / xy) + np.pi/2*(y2x2 > 0)
        alpha = float(alpha)
        #if y2x2 > 0, alpha = alpha + pi/2; end

        a = np.sin(alpha)
        b = np.cos(alpha)
        origin = np.array([meanx,meany])
        self.params = (origin, np.array([a, b]))
        """

        # weighted mean (np.sum(w) = 1)
        origin = (data * w[:, np.newaxis]).sum(axis=0)
        data = data - origin

        # normalize data
        #data, H = musigma_norm(data)
        # general solution
        w = np.diag(w)
        Sw = np.dot(np.dot(np.transpose(data), w), data)
        w, v = np.linalg.eig(Sw)
        normal_vector = v[:,np.argmin(w)]

        self.params = (origin, normal_vector)
        # denormalizes origin and normal_vector
        
        #self.denorm_params(H)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.
        For each point, the shortest (orthogonal) distance to the plane is
        returned. It is obtained by projecting the data onto the plane.
        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `subspace`).
        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        _check_data_atleast_2D(data)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, normal_vector = params
        normal_vector /= np.linalg.norm(normal_vector)
        normal_vector = np.transpose(normal_vector)
        res = (np.dot(data, normal_vector) - np.dot(origin, normal_vector)) / np.linalg.norm(normal_vector)
        return res

    def predict(self, ranges, points_num, seed = 0, params=None):
        """ Builds a hyperplane given the ranges of the subspace
            of the hyperplane.
        Parameters
        ----------
        ranges : (n, m) array
            Coordinate limits along an axis.
        points_num : int
            Number of points along an axis
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).
        Returns
        -------
        data : (n, m) array
            Predicted coordinates.
        """
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')
        
        origin, normal_vector = params
        normal_vector /= np.linalg.norm(normal_vector)

        # Gram–Schmidt orthonormalizing process
        def gs_cofficient(v1, v2):
            return np.dot(v2, v1) / np.dot(v1, v1)

        def multiply(cofficient, v):
          return map((lambda x : x * cofficient), v)

        def proj(v1, v2):
          return multiply(gs_cofficient(v1, v2) , v1)

        def gs(X, row_vecs=True, norm = True):
            if not row_vecs:
                X = X.T
            Y = X[0:1,:].copy()
            for i in range(1, X.shape[0]):
                proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
                Y = np.vstack((Y, X[i,:] - proj.sum(0)))
            if norm:
                Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
            if row_vecs:
                return Y
            else:
                return Y.T

        dim = np.size(normal_vector)
        np.random.seed(int(seed))
        # build hyperplane subspace (based on Gram-Schmidt)
        v = gs(np.array([normal_vector, *(np.random.rand(dim - 1, dim))]))
        subspace = v[1:dim,]

        data = np.zeros((points_num, dim), dtype = float)
        for V, rng in zip(subspace, ranges):
            for coord in data: 
                coord += np.random.uniform(*rng) * V
        data += origin 
        return data

    def set_params(self, coeffs):

        normal_vector = coeffs[:-1]
        normal_vector = np.asarray(normal_vector) / np.linalg.norm(normal_vector)

        dim = np.size(normal_vector)
        w_o = coeffs[dim]
        # some_coord[last_axis] != 0
        first_components = np.zeros(dim - 1)
        last_component = w_o/normal_vector[dim - 1]
        origin = np.append(first_components, np.array(last_component))

        self.params = [origin, normal_vector]
        return True

    def denorm_params(self, H):
        """ Denormalizes general params and
        return denormalized general params"""
        origin, normal_vector = self.params
        dim = np.size(origin)
        origin = np.dot(np.append(origin,1).T, H)

        normal_vector = np.dot(np.append(normal_vector,1).T, H)
        origin = origin/origin[dim]
        normal_vector = normal_vector/normal_vector[dim]
        self.params = (origin[:-1], normal_vector[:-1])
        return True

class CircleModel(BaseModel):

    """Total least squares estimator for 2D circles.
    The functional model of the circle is::
        r**2 = (x - xc)**2 + (y - yc)**2
    This estimator minimizes the squared distances from all points to the
    circle::
        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }
    A minimum number of 3 points is required to solve for the parameters.
    Attributes
    ----------
    params : tuple
        Circle model parameters in the following order `xc`, `yc`, `r`.
    """

    def estimate(self, data):
        """Estimate circle model from data using total least squares.
        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        # http://www.had2know.com/academics/best-fit-circle-least-squares.html
        x2y2 = (x ** 2 + y ** 2)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        m1 = np.stack([[np.sum(x ** 2), sum_xy, sum_x],
                       [sum_xy, np.sum(y ** 2), sum_y],
                       [sum_x, sum_y, float(len(x))]])
        m2 = np.stack([[np.sum(x * x2y2),
                        np.sum(y * x2y2),
                        np.sum(x2y2)]], axis=-1)
        a, b, c = pinv(m1) @ m2
        a, b, c = a[0], b[0], c[0]
        xc = a / 2
        yc = b / 2
        r = np.sqrt(4 * c + a ** 2 + b ** 2) / 2

        self.params = (xc, yc, r)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.
        For each point the shortest distance to the circle is returned.
        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        _check_data_dim(data, dim=2)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params

        xc, yc, r = params

        x = data[:, 0]
        y = data[:, 1]

        return r - np.sqrt((x - xc)**2 + (y - yc)**2)

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.
        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (3, ) array, optional
            Optional custom parameter set.
        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.
        """
        if params is None:
            params = self.params
        xc, yc, r = params

        x = xc + r * np.cos(t)
        y = yc + r * np.sin(t)

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)

class EllipseModel(BaseModel):

    """Total least squares estimator for 2D ellipses.
    The functional model of the ellipse is::
        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)
    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.
    The estimator is based on a least squares minimization. The optimal
    solution is computed directly, no iterations are required. This leads
    to a simple, stable and robust fitting method.
    The ``params`` attribute contains the parameters in the following order::
        xc, yc, a, b, theta
    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`, `b`,
        `theta`.
    params_general : tuple
        Ellipse model general parameters in the following order `a`, `b`, `c`, `d`,
        `f`, `g`.
    """

    def __init__(self):
        BaseModel.__init__(self)
        self.params_general = None

    def estimate(self, data, w = None):
        """Estimate circle model from data using total least squares.
        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).
        """

        #https://es.mathworks.com/matlabcentral/fileexchange/3215-fit_ellipse?s_tid=prof_contriblnk
        # file:///tmp/mozilla_esau0/Weighted_Least_Squares_Fit_of_an_Ellipse_to_Descri.pdf

        # Original Implementation: Ben Hammel, Nick Sullivan-Molina
        # another REFERENCE: [2] http://mathworld.wolfram.com/Ellipse.html
        _check_data_dim(data, dim=2)

        data, H = maxmin_norm(data)

        x = data[:, 0]
        y = data[:, 1]

        # Quadratic part of design matrix [eqn. 15] from [1]
        D1 = np.vstack([x ** 2, x * y, y ** 2]).T
        # Linear part of design matrix [eqn. 16] from [1]
        D2 = np.vstack([x, y, np.ones(len(x))]).T

        if w is None:
            w = np.diag(np.ones(D1.shape[0]))
        else:
            w = np.diag(w)

        # forming scatter matrix [eqn. 17] from [1]
        S1 = D1.T @ w @ D1
        S2 = D1.T @ w @ D2
        S3 = D2.T @ w @ D2

        # Constraint matrix [eqn. 18]
        C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

        try:
            # Reduced scatter matrix [eqn. 29]
            M = inv(C1) @ (S1 - S2 @ inv(S3) @ S2.T)
        except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
            return False

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors
        # from this equation [eqn. 28]
        eig_vals, eig_vecs = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) \
               - np.power(eig_vecs[1, :], 2)
        a1 = eig_vecs[:, (cond > 0)]
        # seeks for empty matrix
        if 0 in a1.shape or len(a1.ravel()) != 3:
            return False
        a, b, c = a1.ravel()

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -inv(S3) @ S2.T @ a1
        d, f, g = a2.ravel()
        
        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
        b /= 2.
        d /= 2.
        f /= 2.

        self.params_general = [a,b,c,d,f,g]
        # denormalizes params_general and sets self.params
        # self.params = denormalized_implicit_params
        self.denorm_params(H)
        
        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.
        For each point the shortest distance to the ellipse is returned.
        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        _check_data_dim(data, dim=2)

        if params is None:
            xc, yc, a, b, theta = self.params
        else:
            xc, yc, a, b, theta = params

        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(t)
            st = math.sin(t)
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt) ** 2 + (yi - yt) ** 2

        # def Dfun(t, xi, yi):
        #     ct = math.cos(t)
        #     st = math.sin(t)
        #     xt = xc + a * ctheta * ct - b * stheta * st
        #     yt = yc + a * stheta * ct + b * ctheta * st
        #     dfx_t = - 2 * (xi - xt) * (- a * ctheta * st
        #                                - b * stheta * ct)
        #     dfy_t = - 2 * (yi - yt) * (- a * stheta * st
        #                                + b * ctheta * ct)
        #     return [dfx_t + dfy_t]

        residuals = np.empty((N, ), dtype=np.double)

        # initial guess for parameter t of closest point on ellipse
        t0 = np.arctan2(y - yc, x - xc) - theta

        # determine shortest distance to ellipse for each point
        for i in range(N):
            xi = x[i]
            yi = y[i]
            # faster without Dfun, because of the python overhead
            t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = np.sqrt(fun(t, xi, yi))

        return residuals

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.
        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5, ) array, optional
            Optional custom parameter set.
        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.
        """

        if params is None:
            params = self.params

        xc, yc, a, b, theta = params

        ct = np.cos(t)
        st = np.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)

    def get_implicit_params(self, params_general = None):

        if params_general is not None:
            a, b, c, d, f, g = params_general
        else:
            a, b, c, d, f, g = self.params_general
        
        # finding center of ellipse [eqn.19 and 20] from [2]
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from [2]
        numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 \
                    - 2 * b * d * f - a * c * g
        term = np.sqrt((a - c) ** 2 + 4 * b ** 2)
        denominator1 = (b ** 2 - a * c) * (term - (a + c))
        denominator2 = (b ** 2 - a * c) * (- term - (a + c))
        width = np.sqrt(2 * numerator / denominator1)
        height = np.sqrt(2 * numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse
        # to x-axis [eqn. 23] from [2].
        phi = 0.5 * np.arctan((2. * b) / (a - c))
        if a > c:
            phi += 0.5 * np.pi
        
        ax = bx = None # 'a' is major axis and 'b' minor axis
        if width >= height:
            ax = width
            bx = height
        else:
            ax = height
            bx = width
            phi -= 0.5 * np.pi

        if phi < 0:
            phi += np.pi

        params = np.nan_to_num([x0, y0, ax, bx, phi]).tolist()
        params = [float(np.real(x)) for x in params]

        return params
    
    def denorm_params(self, H):
        """ Denormalizes general params and
        return implicit params"""
        a, b, c, d, f, g = self.params_general
        coeffs = np.array([ [a, b, d],
                            [b, c, f],
                            [d, f, g]])
        coeffs = np.dot(np.dot(H.T, coeffs), H)
        a, b, c, d, f, g = [coeffs[0,0], coeffs[0,1], coeffs[1,1], coeffs[0,2], coeffs[1,2], coeffs[2,2]]
        self.params_general = (a,b,c,d,f,g)
        self.params = self.get_implicit_params([a,b,c,d,f,g])
        return True

class HomographyModel(BaseModel):

    """Least squares estimator for similarity Homography.
    Similarity Homography is defined by the following matrix:
        Hsim = [[s*cos(phi), -s*sin(phi), tx],
                [s*sin(phi), s*cos(phi),  tx],
                [0,          0,           1]]
    Attributes
    ----------
    params : tuple
        Similarity Homography parameters in the following order:
        `s*cos(phi)`, `s*sin(phi)`, `tx`, `ty`.
    H : (3,3) array
        Similarity homography based on params
    """

    def __init__(self):
        BaseModel.__init__(self)
        self.H = None

    def estimate(self, data, w = None):
        """ Fit homography to selected correspondences
        Parameters
        ----------
        data : (2, N, 2) array
            2 sets of N points in a space of dimensionality dim = 2.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        # target points
        tp = data[:,2:]
        # from points
        fp = data[:,:2]

        if fp.shape != tp.shape:
            raise RuntimeError('number of points or cardinality do not match')
        if fp.shape[0] + tp.shape[0] < 4:
            raise ValueError(f'At least 4 corresponding points are needed - fp.shape: {fp.shape} - tp.shape: {tp.shape}')
            
        fp, tp, denorm_params = self.normalize_points(fp,tp)
        self.params = self.HLS_from_points(fp,tp,w)
        
        #self.params = self.Haffine_from_points(fp,tp)
        #self.set_H()

        # denormalizes similarity homgraphy params
        # and sets the denormalized params self.params = denormalized_params
        # also sets self.H to H based on self.params
        
        self.denormalize_similarity(denorm_params)
        self.set_H()
        return True

    def residuals(self, data, params = None):
        """ Apply homography to all correspondences, 
            return error for each transformed point. """

        def add_ones_column(data):
            """Adds ones column in data
            data.shape is suposed to be (points_num, dim)
            """
            points_num, dim = data.shape
            ones = np.ones((points_num, dim+1))
            ones[:,:-1] = data
            return ones.T

        def normalize(points):
            """ Normalize a collection of points in 
                homogeneous coordinates so that last row = 1. """

            for row in points:
                row /= points[-1]
            return points

        # from points
        tp = add_ones_column(data[:,2:])
        # target points
        fp = add_ones_column(data[:,:2])

        if fp.shape != tp.shape:
            raise RuntimeError('number of points or cardinality do not match')

        if params is not None:
            self.set_H(params)
        H = self.H
        
        # transform fp
        fp_transformed = np.dot(H,fp)
        # normalize hom. coordinates
        fp_transformed = normalize(fp_transformed)
       
        # compute the reprojection error
        residuals = np.sqrt(np.sum((tp-fp_transformed)**2,axis=0))
        return residuals

    def H_from_points(self,fp,tp):
        """ Find homography H, such that fp is mapped to tp
            using the linear DLT method. Points are conditioned
            automatically. NOT UPDATED"""
        
        # create matrix for linear method, 2 rows for each correspondence pair
        nbr_correspondences = fp.shape[1]
        A = zeros((2*nbr_correspondences,9))
        for i in range(nbr_correspondences):        
            A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
                        tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
            A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
                        tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
        
        U,S,V = linalg.svd(A)
        H = V[8].reshape((3,3))    
        
        # decondition
        H = dot(linalg.inv(C2),dot(H,C1))
        
        # normalize and return
        return H / H[2,2]

    def HLS_from_points(self,fp,tp, w = None):
        """ Find homography H, such that fp is mapped to tp
            using the linear DLT method. A particular solution 
            to find similarity homography is used.
        Parameters
        ----------
        fp : (N, 2) array
            N `from points` in a space of dimensionality dim = 2.
        tp : (N, 2) array
            N `to points` in a space of dimensionality dim = 2.
        Returns
        -------
        params : tuple
            Params in the following order `s*cos(phi)`, `s*sin(phi)`, `tx`, `ty`.
        """

        if w is None:
            w = np.ones((tp.shape[0],))
        # see simestimator.pdf for detailed derivation
        
        N = tp.shape[0]
        A = np.sum(w**2*(np.sum(fp**2, axis = -1)))
        B = np.sum(w*fp[:,0])
        C = np.sum(w*fp[:,1])

        # ((M'W'W)Q)'= [D, E, F, G]
        D = np.array([fp[:,0]*tp[:,0], fp[:,1]*tp[:,1]])
        D = np.sum(w**2*(np.sum(D.T, axis = -1)))
        
        E = np.array([fp[:,0]*tp[:,1], -fp[:,1]*tp[:,0]])
        E = np.sum(w**2*(np.sum(E.T, axis = -1)))
        
        F = np.sum(w*tp[:,0])
        G = np.sum(w*tp[:,1])

        # solving Ô = [s_cos_phi, s_sin_phi, tx, ty]
        #den = B**2 + C**2 - A*N
        #s_cos_phi = B*F + C*G - D*N / den
        #s_sin_phi = B*G - C*F - E*N / den
        #tx = B*D - A*F - C*E / den
        #ty = B*E + C*D - A*G / den
        
        # considering mu_tp = (0,0) and mu_fp = (0,0)
        s_cos_phi = D/A
        s_sin_phi = E/A
        tx = 0
        ty = 0
        """
        cx, sx, xx = [0, 0, 0]

        for P, Q, w_i in zip(fp, tp, w):        
            cx += (P[0] * Q[0] + P[1] * Q[1])*w_i
            sx += (P[0] * Q[1] - P[1] * Q[0])*w_i
            xx += (P[0] * P[0] + P[1] * P[1])*w_i

        s_cos_phi = cx / xx
        s_sin_phi = sx / xx
        tx = 0
        ty = 0
        """
        return [s_cos_phi, s_sin_phi, tx, ty]

    def Hsim_from_points(self,fp,tp, w = None):
        """ Find homography H, such that fp is mapped to tp
            using LS. A particular solution to find similarity 
            homography is used.
        Parameters
        ----------
        fp : (2, 2) array
            2 `from points` in a space of dimensionality dim = 2.
        tp : (2, 2) array
            2 `to points` in a space of dimensionality dim = 2.
        Returns
        -------
        params : tuple
            Params in the following order `s*cos(phi)`, `s*sin(phi)`, `tx`, `ty`.
        """

        fp1, fp2 = [fp[0],fp[1]]
        tp1, tp2 = [tp[0],tp[1]]
        dfp = fp1 - fp2
        dtp = tp1 - tp2

        den = dfp[0]**2 + dfp[1]**2
        s_cos_phi = (dfp[0] * dtp[0] + dfp[1] * dtp[1]) / den
        s_sin_phi = (dfp[0] * dtp[1] - dfp[1] * dtp[0]) / den
        tx = tp2[0] - fp2[0] * s_cos_phi + fp2[1] * s_sin_phi
        ty = tp2[1] - fp2[0] * s_sin_phi - fp2[1] * s_cos_phi

        return [s_cos_phi, s_sin_phi, tx, ty]

    def Haffine_from_points(self, fp,tp):
        """ Find H, affine transformation, such that 
            tp is affine transf of fp. NOT UPDATED"""
        
        ones = np.ones((fp.shape[0],))
        fp = np.vstack((fp.T,ones))
        tp = np.vstack((tp.T,ones))

        if fp.shape != tp.shape:
            raise RuntimeError('number of points do not match')
            
        # condition points
        # --from points--
        m = np.mean(fp[:2], axis=1)
        maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
        C1 = np.diag([1/maxstd, 1/maxstd, 1]) 
        C1[0][2] = -m[0]/maxstd
        C1[1][2] = -m[1]/maxstd
        fp_cond = np.dot(C1,fp)
        
        # --to points--
        m = np.mean(tp[:2], axis=1)
        C2 = C1.copy() #must use same scaling for both point sets
        C2[0][2] = -m[0]/maxstd
        C2[1][2] = -m[1]/maxstd
        tp_cond = np.dot(C2,tp)
        
        # conditioned points have mean zero, so translation is zero
        A = np.concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
        U,S,V = np.linalg.svd(A.T)
        
        # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
        tmp = V[:2].T
        B = tmp[:2]
        C = tmp[2:4]
        
        tmp2 = np.concatenate((np.dot(C,np.linalg.pinv(B)),np.zeros((2,1))), axis=1) 
        H = np.vstack((tmp2,[0,0,1]))
       
        # decondition
        H = np.dot(np.linalg.inv(C2),np.dot(H,C1))

        myH = np.array(H/H[2,2])
        myH = myH.reshape(3,3)
        s_cos_phi = myH[0,0]
        s_sin_phi = myH[1,0]
        tx = myH[1,0]
        ty = myH[1,1]

        return [s_cos_phi, s_sin_phi, tx, ty]
        #return H / H[2,2]
    
    def normalize_points(self,fp,tp):
        """ Normalize points (normalization proposed 
        in "Multiple View Geometry in Computer Vision" by Hartley - Zisserman)
        Parameters
        ----------
        fp : (N, 2) array
            N `from points` in a space of dimensionality dim = 2.
        tp : (N, 2) array
            N `to points` in a space of dimensionality dim = 2.
        Returns
        -------
        fp : (N, 2) array
            Normalized `from points`
        tp : (N, 2) array
            Normalized `to points`
        denorm_params: tuple
            Denormalization params to denormalize similarity homography
        """
        n = fp.shape[0]
        Pc = np.mean(fp, axis = 0)
        Qc = np.mean(tp, axis = 0)

        Ps = np.sum(np.abs(fp - Pc)**2, axis = -1)
        Qs = np.sum(np.abs(tp - Qc)**2, axis = -1)
        
        sqrt2_n = np.sqrt(2.0)*n
        
        Ps = sqrt2_n/np.sum(np.sqrt(Ps))
        Qs = sqrt2_n/np.sum(np.sqrt(Qs))

        return (fp-Pc)*Ps, (tp-Qc)*Qs, [Ps, Qs, Pc, Qc]

    def denormalize_similarity(self, denorm_params):
        """ Denormlizes similarity matrix based on 
        denormalization params: scale and centroid
        Parameters
        ----------
        denorm_params: tuple
            Denormalization params in the following order
            denorm_params = [Ps, Qs, Pc, Qc]
        Returns
        -------
        success : bool
            True, if denormalization succeeds.
        """

        Ps, Qs, Pc, Qc = denorm_params
        s_cos_phi, s_sin_phi, tx, ty = self.params
        
        scale_ratio = Ps/Qs

        s_cos_phi *= scale_ratio
        s_sin_phi *= scale_ratio
        tx = Qc[0] + tx/Qs - (Pc[0] * s_cos_phi - Pc[1] * s_sin_phi)
        ty = Qc[1] + ty/Qs - (Pc[0] * s_sin_phi + Pc[1] * s_cos_phi)

        self.params = (s_cos_phi, s_sin_phi, tx, ty)
        self.set_H()
        
        return True

    def set_H(self, params = None):
        """ Sets self.H based on self.params 
        Parameters
        ----------
        params (optional): tuple
            Params in the following order `s*cos(phi)`, `s*sin(phi)`, `tx`, `ty`
        Returns
        -------
        success : bool
            True, if setting succeeds.
        """
        if params is not None:
            s_cos_phi, s_sin_phi, tx, ty = params
        else:
            s_cos_phi, s_sin_phi, tx, ty = self.params

        self.H = [  [s_cos_phi, -s_sin_phi, tx],
                    [s_sin_phi, s_cos_phi,  ty],
                    [0,         0,          1]]

        return True

    def get_projection(self, data):
        """ returns projection x' = Hx
        """
        N, dim = data.shape
        ones = np.ones((N, dim+1))
        ones[:,:-1] = data
        data = np.dot(self.H, ones.T)
        for row in data:
            row /= data[-1]
            print(np.sum(row))
        return data[:,:-1].T
