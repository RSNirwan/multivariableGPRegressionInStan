functions{
    vector gp_pred_rng(vector[] X, vector[] X_pred, vector y, 
                    real kernel_std, real kernel_lengthscale, real noise_std){
        // returns a sample from posterior predictive distribution
        int N = size(X);
        int N_pred = size(X_pred);
        vector[N_pred] y_pred;
        {
            matrix[N,N] K;
            //cholesky_factor_cov[N] L;          // cholesky of K(X,X)
            matrix[N,N] L;
            matrix[N_pred,N] K_x_pred_x;
            matrix[N_pred,N_pred] K_x_pred_x_pred;
            matrix[N, N_pred] L_div_K_x_x_pred;
            vector[N] K_div_y;
            vector[N_pred] y_pred_mean; 
            matrix[N_pred, N_pred] y_pred_cov;
            
            K = cov_exp_quad(X, kernel_std, kernel_lengthscale);
            K_x_pred_x = cov_exp_quad(X_pred, X, kernel_std, kernel_lengthscale);
            for (n in 1:N)
                K[n,n] = K[n,n] + square(noise_std) + 1e-14;
            L = cholesky_decompose(K);
            
            //y_pred_mean = K_x_pred_x * inv(K) * y
            K_div_y = mdivide_left_tri_low(L, y);
            K_div_y = mdivide_right_tri_low(K_div_y', L)';
            y_pred_mean = K_x_pred_x * K_div_y;
            
            //y_pred_std = sqrt(diag( K_x_pred_x_pred - K_x_pred_x*inv(K)*K_x_pred_x' ))
            L_div_K_x_x_pred = mdivide_left_tri_low(L, K_x_pred_x');
            K_x_pred_x_pred = cov_exp_quad(X_pred, kernel_std, kernel_lengthscale);
            y_pred_cov = K_x_pred_x_pred - L_div_K_x_x_pred' * L_div_K_x_x_pred;

            y_pred = multi_normal_rng(y_pred_mean, 
                                  y_pred_cov + diag_matrix(rep_vector(1e-14, N_pred)));
        }
        return y_pred;
    }
}
data{
    int<lower=1> N;
    int<lower=1> N_pred;
    int<lower=1> D;
    vector[N] y;
    vector[D] X[N];
    vector[D] X_pred[N_pred];
}
transformed data {
    vector[N] mu = rep_vector(0, N);
}
parameters{
    real<lower=0> kernel_lengthscale;
    real<lower=0> kernel_std;
    real<lower=0> noise_std; 
}
transformed parameters{
    cholesky_factor_cov[N] L;
    {
        matrix[N,N] K = cov_exp_quad(X, kernel_std, kernel_lengthscale);
        for (n in 1:N)
            K[n,n] = K[n,n] + square(noise_std) + 1e-14;
        L = cholesky_decompose(K);
    }
}
model{
    kernel_std ~ normal(0, 1);
    kernel_lengthscale ~ inv_gamma(2.0,1.0);
    noise_std ~ normal(0, 1);
    
    y ~ multi_normal_cholesky(mu, L);
}
generated quantities{
    vector[N_pred] y_pred = gp_pred_rng(X, X_pred, y, kernel_std, kernel_lengthscale, noise_std);
}



