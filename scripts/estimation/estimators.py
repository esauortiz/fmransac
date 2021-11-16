import numpy as np

class RANSAC(object):
    def __init__(   self, min_samples, residual_threshold, 
                    max_trials, stop_probability, 
                    outlier_ratio = None):

        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.stop_probability = stop_probability
        self.outlier_ratio = outlier_ratio

    def _teoric_max_trials(self):
        """Determine number trials such that at least one outlier-free subset is
        sampled for the given inlier/outlier ratio.
        Parameters
        ----------
        min_samples : int
            Minimum number of samples chosen randomly from original data.
        probability : float
            Probability (confidence) that one outlier-free sample is generated.
            Desired probability that we get a good sample
        outlier_ratio : float
        Returns
        -------
        trials : int
            Number of trials.
        """
        nom = 1 - self.stop_probability
        if nom == 0:
            return np.inf

        inlier_ratio = 1 - self.outlier_ratio

        denom = 1 - inlier_ratio ** self.min_samples
        if denom == 0:
            return 1
        elif denom == 1:
            return np.inf

        nom = np.log(nom)
        denom = np.log(denom)
        if denom == 0:
            return 0

        return int(np.ceil(nom / denom))

    def _loss_function(self, residuals):
        """Determine base ransac version score function
        Parameters
        ----------
        self.residual_threshold : float
            residual threshold, determines if a point is an outlier
        residuals :	array(dtype = float)
            array of residuals w.r.t. sample model
        """
        return abs(residuals) > self.residual_threshold

    def run(self, data, model_class, seed = None):
        """ Core of RANSAC algorithm
        Parameters
        ----------
        data : (M, N) array
            array where each sample is placed as row
        model_class : class
            model class to be estimated. It is mandatory model
            class to have a 'residuals' class method.
        """
        
        # best estimation variables
        best_model = None
        best_inliers = np.zeros((np.shape(data)[0],), dtype = bool)
        best_residuals = np.zeros((np.shape(data)[0],), dtype = float)
        best_scores = np.ones((np.shape(data)[0],), dtype = float) * np.inf
        seed = int(seed)

        # random variable to choose minimal sample sets
        random_state = np.random.RandomState(seed)
        
        teoric_max_trials = self._teoric_max_trials()
        if self.max_trials > teoric_max_trials: self.max_trials = teoric_max_trials

        if not isinstance(data, (tuple, list)):
                data = (data, )

        num_samples = len(data[0])

        if not (0 < self.min_samples < num_samples):
            raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

        if self.residual_threshold < 0:
            raise ValueError("`residual_threshold` must be greater than zero")

        if self.max_trials < 0:
            raise ValueError("`max_trials` must be greater than zero")

        if not (0 <= self.stop_probability <= 1):
            raise ValueError("`stop_probability` must be in range [0, 1]")

        #for the first run use initial guess of inliers
        spl_idxs = random_state.choice(num_samples, self.min_samples, replace=False)
        
        for num_trials in range(self.max_trials):
            #do sample selection according data pairs
            samples = [d[spl_idxs] for d in data]
            #for next iteration choose random sample set and be sure that no samples repeat
            spl_idxs = random_state.choice(num_samples, self.min_samples, replace=False)
            # estimate model for current random sample set
            sample_model = model_class()

            # First estimation i.e. whithout weights
            success = sample_model.estimate(*samples)
            # if the model could not be estimate then continue
            if success is not None and not success:
                continue
            sample_model_residuals = np.abs(sample_model.residuals(*data))
            # compute scores
            sample_model_scores = self._loss_function(sample_model_residuals)
            # consensus set / inliers
            sample_model_inliers = sample_model_residuals < self.residual_threshold
            sample_model_score = np.sum(sample_model_scores)

            best_score = np.sum(best_scores)
            if (best_score > sample_model_score):
                best_model = sample_model
                best_scores = sample_model_scores
                best_inliers = sample_model_inliers
                best_residuals = sample_model_residuals

                #dynamic_max_trials = _max_trials(np.sum(best_inliers), num_samples, min_samples, stop_probability)
                #if num_trials >= dynamic_max_trials:
                #	break
        
        #np.savetxt(f'/home/esau/tfm/slides/mss/G{self.max_trials}.txt',*samples)
        #return sample_model, sample_model_inliers, sample_model_scores, 0
        
        # failure if inliers_num < min_samples
        if np.sum(best_inliers) < self.min_samples:
            best_model = None
            best_inliers = None
            best_scores = None
            return best_model, best_inliers, best_scores, num_trials

        # estimate final model using all inliers
        if best_inliers is not None:
            # select inliers for each data array (variant_1 and ransac)
            data_inliers = [d[best_inliers] for d in data]
            best_model.estimate(*data_inliers)

        #best_residuals = best_model.residuals(*data)
        return best_model, best_inliers, best_scores, 0

class MSAC(RANSAC):
    def _loss_function(self, residuals):
        """Determine MSAC score function
        Parameters
        ----------
        threshold : float
            residual threshold, determines if a point is an outlier
        residuals :	array(dtype = float)
            array of residuals w.r.t. sample model
        """	
        size = np.size(residuals)
        scores = np.ones(size) * (self.residual_threshold ** 2)
        for i, residual in zip(range(size), residuals):
            if residual > self.residual_threshold:
                continue
            scores[i] = residual**2
        return scores

class FMR(RANSAC):
    # scores and compatibility values mean the same
    def __init__(   self, min_samples, residual_threshold, 
                    max_trials, stop_probability, variant, fuzzy_metric, t_max = 1, sigma_phi = 0, 
                    convergence_threshold = 0.0005, outlier_ratio = None):
        RANSAC.__init__(self, min_samples, residual_threshold, 
                        max_trials, stop_probability, outlier_ratio)
        
        # variant of FMR
        self.variant = variant
        self.fuzzy_metric = fuzzy_metric
        # number of iterations at _iterative_reestimation stage
        #  if t_max = 1 only one reestimation will be performed
        self.t_max = t_max
        # parameter of variant 3
        self.sigma_phi = sigma_phi
        # convergence threshold for _iterative_reestimation stage
        self.convergence_threshold = convergence_threshold

    def _score_function(self, residuals):
        return self.fuzzy_metric.compatibilities(residuals)

    def _iterative_reestimation(self, data, model_class, model, scores,
                                best_inliers = None, iterations = 0,
                                test_id = None, save_path = None, # debug variables
                                file_name = None, debug = False): # debug variables

        if iterations is not None:
            iterations += 1

        # prev_model and new_model
        prev_params = model.params
        new_model = model_class()

        # normalize scores and estimate new_model

        if best_inliers is None:
            scores = scores / np.sum(scores)
            new_model.estimate(*data, scores)
        else:
            data_inliers = [d[best_inliers] for d in data]
            scores_inliers = scores[best_inliers]
            scores_inliers = scores_inliers / np.sum(scores_inliers)
            new_model.estimate(*data_inliers, scores_inliers)

        # new residuals and scores
        new_residuals = new_model.residuals(*data)
        new_scores = self._score_function(np.abs(new_residuals))

        def _check_convergence(prev_params, new_params):
            if np.max(abs(np.asarray(prev_params) - np.asarray(new_params))) < self.convergence_threshold:
                return True
            return False

        if (_check_convergence(prev_params, new_model.params) == True) or iterations >= self.t_max:
            return new_model, new_scores, iterations
        else:
            return self._iterative_reestimation(data, model_class, new_model,
                                                new_scores, best_inliers, iterations, 
                                                test_id, save_path, file_name, debug)

    def run(self, data, model_class, seed = None):
        """ Core of Fuzzy Metric Based RANSAC algorithm
        Parameters
        ----------
        data : (M, N) array
            array where each sample is placed as row
        model_class : class
            model class to be estimated. It is mandatory model
            class to have a 'residuals' class method.
        fuzzy_metric : FuzzyMetric class
            fuzzy metric to compute samples compatibilities
        """

        # best estimation variables
        best_model = None
        best_inliers = np.zeros((np.shape(data)[0],), dtype = bool)
        best_residuals = np.zeros((np.shape(data)[0],), dtype = float)
        best_scores = np.zeros((np.shape(data)[0],), dtype = float)
        seed = int(seed)

        if self.variant == 4:
            # all points are considered inliers
            sample_model_inliers = np.ones((np.shape(data)[0],), dtype = bool)

        # random variable to choose minimal sample sets
        random_state = np.random.RandomState(seed)
        
        teoric_max_trials = self._teoric_max_trials()
        if self.max_trials > teoric_max_trials: self.max_trials = teoric_max_trials

        if not isinstance(data, (tuple, list)):
                data = (data, )

        num_samples = len(data[0])

        if not (0 < self.min_samples < num_samples):
            raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

        if self.residual_threshold < 0:
            raise ValueError("`residual_threshold` must be greater than zero")

        if self.max_trials < 0:
            raise ValueError("`max_trials` must be greater than zero")

        if not (0 <= self.stop_probability <= 1):
            raise ValueError("`stop_probability` must be in range [0, 1]")

        #for the first run use initial guess of inliers
        spl_idxs = random_state.choice(num_samples, self.min_samples, replace=False)

        for num_trials in range(self.max_trials):
            #do sample selection according data pairs
            samples = [d[spl_idxs] for d in data]
            #for next iteration choose random sample set and be sure that no samples repeat
            spl_idxs = random_state.choice(num_samples, self.min_samples, replace=False)
            # estimate model for current random sample set
            sample_model = model_class()

            # First estimation i.e. whitout weights
            success = sample_model.estimate(*samples)
            # if the model could not be estimate then continue
            if success is not None and not success:
                continue
            sample_model_residuals = np.abs(sample_model.residuals(*data))
            # compute scores
            sample_model_scores = self._score_function(sample_model_residuals)

            # consensus set / inliers
            if self.variant in [1, 2]:
                sample_model_inliers = sample_model_residuals < self.residual_threshold
                # only taking into account inliers scores
                sample_model_scores[sample_model_inliers == False] = 0
                sample_model_score = np.sum(sample_model_scores)
            elif self.variant == 3:
                sample_model_inliers = sample_model_scores > self.sigma_phi
                # only taking into account inliers scores
                sample_model_scores[sample_model_inliers == False] = 0
                sample_model_score = np.sum(sample_model_scores)
            elif self.variant == 4:
                sample_model_score = np.sum(sample_model_scores)

            best_score = np.sum(best_scores)
            if best_score < sample_model_score:
                best_model = sample_model
                best_scores = sample_model_scores
                best_inliers = sample_model_inliers
                best_residuals = sample_model_residuals

                #dynamic_max_trials = _max_trials(np.sum(best_inliers), num_samples, min_samples, stop_probability)
                #if num_trials >= dynamic_max_trials:
                #	break
        
        #np.savetxt(f'/home/esau/tfm/slides/mss/{self.__class__.__name__}{self.variant}_{self.fuzzy_metric.__class__.__name__}/G{self.max_trials}.txt',*samples)
        #return sample_model, sample_model_inliers, sample_model_scores, 0
        
        # failure if inliers_num < min_samples
        if np.sum(best_inliers) < self.min_samples:
            best_model = None
            best_inliers = None
            best_scores = None
            improvements = None
            return best_model, best_inliers, best_scores, improvements

        # estimate final model using all inliers
        weights = np.copy(best_scores)
        if best_inliers is not None and self.t_max == 1:
            iterations = 1
            if self.variant == 1:
                # select inliers
                data_inliers = [d[best_inliers] for d in data]
                best_model.estimate(*data_inliers)
            else:
                # weighted estimation
                if self.variant == 4:
                    weights = weights / np.sum(weights)
                    best_model.estimate(*data, weights)
                else:
                    data_inliers = [d[best_inliers] for d in data]
                    weights_inliers = weights[best_inliers]
                    weights_inliers = weights_inliers / np.sum(weights_inliers)
                    best_model.estimate(*data_inliers, weights_inliers)


        # iterative reestimation with at most t_max iterations
        elif best_inliers is not None and self.t_max > 1:
            iterations = 0
            if self.variant in [2, 3]:
                # best_inliers as argument for first iteration
                best_model, best_scores, iterations = self._iterative_reestimation( data, model_class, best_model, weights, 
                                                                                    best_inliers = best_inliers, iterations = 0)
            elif self.variant == 4:
                # best_inliers are not specified because whole dataset is used
                best_model, best_scores, iterations = self._iterative_reestimation( data, model_class, best_model, 
                                                                                    weights, iterations = 0)                
        # update best model scores
        best_scores = self._score_function(np.abs(best_model.residuals(*data)))
        return best_model, best_inliers, best_scores, iterations