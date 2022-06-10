# date: 11.29.2020
# memory-optimized + more efficient gradient

# date: 02.09, 2021
# heartsteps data analysis - incorporate the availabiity


cv <- function(dat, policy_class=NULL, nfold = 3, type_constraint=1){
  
  
  ##### helper function ####
  k0 = function(z1, x1, z2, x2, sigma = 1) exp(-sum((x1-x2)^2) * sigma)
  
  kernel_train = function(X, Z, A, sigma = 1){
    
    
    
    X <- cbind(X, Z)
    
    temp <- (2 * A - 1) %*% t(2 * A - 1)
    
    #temp[temp == -1] <- 0
    
    #   N <- nrow(X)
    #   
    #   DotProduct <- X %*% t(X)
    #   
    #   DiagDotProduct = as.matrix(diag(DotProduct)) %*% t(as.matrix(rep(1,N)))
    #   
    #   KernelMatrix = DiagDotProduct + t(DiagDotProduct) - 2*DotProduct
    #   
    #   
    #   
    #   KernelMatrix  = exp(-KernelMatrix)
    
    
    
    #median heuristic to select bandwidth
    
    
    
    rbf <- rbfdot(sigma = sigma)
    KK <- kernelMatrix(rbf, X)
    
    
    KK[temp==-1] <- 0
    return(KK)
    
    
  }
  
  kernel_test = function(X, Y, sigma = 1){
    
    N1 <- nrow(X)
    
    N2 <- nrow(Y)
    
    A1 <- X[, ncol(X)]
    A2 <- Y[, ncol(Y)]
    
    X <- X[, 1:(ncol(X)-1)]
    Y <- Y[, 1:(ncol(Y)-1)]
    
    temp <- (2 * A1 - 1) %*% t(2 * A2 - 1)
    
    #temp[temp == -1] <- 0
    #   
    #   DiagDotProductX <- replicate(N2, apply(X * X, 1, sum))
    #   DiagDotProductY1 <- t(replicate(N1, apply(Y * Y, 1, sum)))
    #   DotProductXY1 <- X %*% t(Y)
    #   KK1 <- exp(- (DiagDotProductX + DiagDotProductY1 - 2 * DotProductXY1))  
    #   
    
    rbf <- rbfdot(sigma = sigma)
    KK1 <- kernelMatrix(rbf, X, Y)
    
    
    KK1[temp==-1] <- 0
    
    KK1 <- t(KK1)
    
    
    
    
    return(KK1)                    
  }
  
  shift.Q <-  function(X, Z, A, sigma = 1) {
    
    
    
    
    X1 <- cbind(X, Z, A)
    N <- length(A)
    X2 <- cbind(t(replicate(N, x.star)), Z, a.star)
    
    output <- kernel_train(X, Z, A, sigma) - t(kernel_test(X1, X2, sigma)) - 
      t(kernel_test(X2, X1, sigma)) + t(kernel_test(X2, X2, sigma))
    
    return(output)
    
    
  }
  
  testshift.Q <-  function(X1, Z1, A1, X2, Z2, A2, sigma = 1) {
    
    
    #   k1(z1, x1, a1, z2, x2, a2) - 
    #     k1(z1, x.star, a.star, z2, x2, a2) - 
    #     k1(z1, x1, a1, z2, x.star, a.star) + 
    #     k1(z1, x.star, a.star, z2, x.star, a.star)  
    #   
    
    X3 <- cbind(X1, Z1, A1)
    X4 <- cbind(X2, Z2, A2)
    
    N1 <- length(A1)
    X5 <- cbind(t(replicate(N1, x.star)), Z1, a.star)
    
    N2 <- length(A2)
    X6 <- cbind(t(replicate(N2, x.star)), Z2, a.star)
    
    
    
    output <- t(kernel_test(X3, X4, sigma)) - t(kernel_test(X5, X4, sigma)) -
      t(kernel_test(X3, X6, sigma)) + t(kernel_test(X5, X6, sigma))
    
    return(output)
    
    
  }
  
  
  
  
  
  # Function to calculate shifted kernal
  kernel.Q = function(z1, x1, a1, z2, x2, a2) {
    
    k1(z1, x1, a1, z2, x2, a2) - 
      k1(z1, x.star, a.star, z2, x2, a2) - 
      k1(z1, x1, a1, z2, x.star, a.star) + 
      k1(z1, x.star, a.star, z2, x.star, a.star)  
    
    
  }
  
  # number of baseline covariates Z
  phi = function(z) 1
  
  # kernel between any two cases
  k1 = function(z1, x1, a1, z2, x2, a2, sigma = 1) (a1==a2) * k0(z1, x1, z2, x2, sigma)
  
  
  
  
  # Given the full data, calculate K, L and Phi
  pre_CalMat = function(dat, nZ, nX){
    
    
    
    
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    
    sigma <- 1 / (median(dist(X)))^2
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    
    
    S1 <- t(testshift.Q(X, Z,A, X, Z, matrix(1, N, 1), sigma = sigma))
    S2 <- t(testshift.Q(X, Z,A, X, Z, matrix(0, N, 1),sigma = sigma))
    
    
    
    
    TT <- shift.Q(X, Z, A, sigma = sigma)
    L  <- kernel_train(X, Z, A, sigma = sigma)
    
    
    temp1 <- testshift.Q(X.next, Z, matrix(1, N, 1), X, Z, A, sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d1 <- replicate(N, c(prob1)) * temp1
    
    
    temp2 <- testshift.Q(X.next, Z, matrix(0, N, 1), X, Z, A, sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d2 <- replicate(N, c(1-prob1)) * temp2
    
    
    temp3 <- testshift.Q(X, Z, A, X.next, Z, matrix(1, N, 1), sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d3 <- t(replicate(N, c(prob1))) * (temp3)
    
    
    temp4 <- testshift.Q(X, Z, A, X.next, Z, matrix(0, N, 1), sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d4 <- t(replicate(N, c(1-prob1))) * (temp4)
    
    
    temp5 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(1, N, 1), sigma = sigma)
    
    #d5 <- (prob1 %*% t(prob1)) * temp5 
    
    temp6 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(0, N, 1), sigma = sigma)
    
    #d6 <- (prob1 %*% t(1 - prob1)) * temp6 
    
    temp7 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(1, N, 1), sigma = sigma)
    
    #d7 <- ((1-prob1) %*% t(prob1)) * temp7
    
    temp8 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(0, N, 1), sigma = sigma)
    
    #d8 <- ((1-prob1) %*% t(1-prob1)) * temp8
    
    #K <- TT - (d1 + d2 + d3 + d4) + d5 + d6 + d7 + d8
    
    
    
    list(L = L, Phi = Phi, TT = TT, S1 = S1, S2 = S2,
         temp1 = temp1, temp2 = temp2,temp3 = temp3,temp4 = temp4,
         temp5 = temp5,temp6 = temp6,temp7 = temp7,temp8 = temp8)
    
  }
  
  
  
  
  
  
  
  CalMat = function(dat, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    
    
    
    # temp1 <- trn$temp1
    # temp2 <- trn$temp2
    # temp3 <- trn$temp3
    # temp4 <- trn$temp4
    # temp5 <- trn$temp5
    # temp6 <- trn$temp6
    # temp7 <- trn$temp7
    # temp8 <- trn$temp8
    # L <- trn$L
    # TT <- trn$TT
    # S1 <- trn$S1
    # S2 <- trn$S2
    
    
    
    ########################expand x using policy class###############################
    
    if(p == 2){
      
      policy_X      <- matrix(apply(X, 1, policy_class), ncol = 1)
      policy_X.next <- matrix(apply(X.next, 1, policy_class), ncol = 1)
      
      
    }else{
      
      policy_X      <- t(apply(X, 1, policy_class))
      policy_X.next <- t(apply(X.next, 1, policy_class))
      
      
    }
    
    
    ###############################################################################
    
    
    prob1 <- 1 / (1 + exp(-cbind(Z, policy_X.next) %*% theta))
    
    
    d1 <- replicate(N, c(prob1)) * trn$temp1
    
    
    #temp2 <- testshift.Q(X.next, Z, matrix(0, N, 1), X, Z, A)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d2 <- replicate(N, c(1-prob1)) * trn$temp2
    
    
    #temp3 <- testshift.Q(X, Z, A, X.next, Z, matrix(1, N, 1))
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d3 <- t(replicate(N, c(prob1))) * (trn$temp3)
    
    
    #temp4 <- testshift.Q(X, Z, A, X.next, Z, matrix(0, N, 1))
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d4 <- t(replicate(N, c(1-prob1))) * (trn$temp4)
    
    
    #temp5 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(1, N, 1))
    
    d5 <- (prob1 %*% t(prob1)) * trn$temp5 
    
    #temp6 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(0, N, 1))
    
    d6 <- (prob1 %*% t(1 - prob1)) * trn$temp6 
    
    #temp7 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(1, N, 1))
    
    d7 <- ((1-prob1) %*% t(prob1)) * trn$temp7
    
    #temp8 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(0, N, 1))
    
    d8 <- ((1-prob1) %*% t(1-prob1)) * trn$temp8
    
    K <- trn$TT - (d1 + d2 + d3 + d4) + d5 + d6 + d7 + d8
    
    
    
    #   list(L = L, K = K, Phi = Phi, TT = TT, S1 = S1, S2 = S2,
    #        temp1 = temp1, temp2 = temp2,temp3 = temp3,temp4 = temp4,
    #        temp5 = temp5,temp6 = temp6,temp7 = temp7,temp8 = temp8)
    list(K = K)
  }
  
  deriv_of_K = function(dat, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    # temp1 <- trn$temp1
    # temp2 <- trn$temp2
    # temp3 <- trn$temp3
    # temp4 <- trn$temp4
    # temp5 <- trn$temp5
    # temp6 <- trn$temp6
    # temp7 <- trn$temp7
    # temp8 <- trn$temp8
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    p     <- length(theta)
    
    
    
    deriv_KK <- array(0, dim = c(N, N, p))
    
    ########################expand x using policy class###############################
    
    
    if(p == 2){
      
      policy_X      <- matrix(apply(X, 1, policy_class), ncol = 1)
      policy_X.next <- matrix(apply(X.next, 1, policy_class), ncol = 1)
      
      
    }else{
      
      policy_X      <- t(apply(X, 1, policy_class))
      policy_X.next <- t(apply(X.next, 1, policy_class))
      
      
    }
    
    ###############################################################################
    
    
    prob1 <- 1 / (1 + exp(-cbind(Z, policy_X.next) %*% theta))
    
    d3 <- t(replicate(N, c(prob1))) * (trn$temp3)
    
    d4 <- t(replicate(N, c(1-prob1))) * (trn$temp4)
    
    
    ###################################### we use the policy class
    design_Xnext <- cbind(Z, policy_X.next)
    
    ########################################
    
    
    for(k in 1:p){
      
      
      dd1 <- replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k]))
      
      dd2 <- -replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k]))
      
      dd3 <- t(replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k])))
      
      dd4 <- -t(replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k])))
      
      dd5 <- (prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(prob1) +
        (prob1) %*% t(prob1* (1 - prob1) * design_Xnext[, k])
      
      dd6 <- (prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(1 - prob1) +
        (prob1) %*% t(-prob1* (1 - prob1) * design_Xnext[, k])
      
      dd7 <- (- prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(prob1) +
        (1 - prob1) %*% t(prob1* (1 - prob1) * design_Xnext[, k])
      
      dd8 <- (- prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(1 - prob1) +
        (1 - prob1) %*% t(- prob1* (1 - prob1) * design_Xnext[, k])
      
      
      
      
      
      deriv_KK[, , k] = -(dd1 * trn$temp1 + dd2 * trn$temp2) +
        -(dd3 * trn$temp3 + dd4 * trn$temp4) +
        dd5 * trn$temp5 + dd6 * trn$temp6 +
        dd7 * trn$temp7 + dd8 * trn$temp8 
      
    }
    
    
    return(deriv_KK)
    
  }
  
  
  obtainTD.err = function(cross.K, tst.dat, alpha, beta){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    if(nZ == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    
    TD.err <- c(tst.dat$reward - Phi %*% beta - t(cross.K) %*% alpha)
    
    
    
    return(TD.err)
    
  }
  
  
  obtainTD.err.ratio = function(cross.K, tst.dat, alpha){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    if(nZ == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    response <- rep(1, length(tst.dat$reward))
    TD.err <- c(response - t(cross.K) %*% alpha)
    
    
    
    return(TD.err)
    
  }
  
  
  valid = function(TD.err, tst.dat, nZ, nX){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(tst.dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    A <- tst.dat$action;
    
    
    m1 <- gausspr(TD.err[A==1], X[A==1,], verbose = F)
    m0 <- gausspr(TD.err[A==0], X[A==0,], verbose = F)
    
    (mean((predict(m1))^2) + mean((predict(m0))^2))/2 # mean squared estimated Bel.Err
    
  }
  
  
  
  
  ###Rregret estimation
  
  RegEst = function(dat, lambda, mu, lambda2, mu2, nZ, nX,
                    theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      L <- trn$L
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      temp <- solve(L+ mu*N*diag(N), L)
      trn$M <- tcrossprod(temp)
      
      temp <- solve(L+ mu2*N*diag(N), L)
      trn$M2 <- tcrossprod(temp)
      
      
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      # M2 <- trn$M2
      
    }
    
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    # estimates (depending on tuning parameter)
    #     temp <- solve(L+ mu*N*diag(N), L)
    #     M <- tcrossprod(temp)
    
    term1 <- trn$M %*% pre$K
    
    term2 <- (trn$M %*% Phi) 
    
    term3 <- trn$M %*% R
    
    
    
    
    #AA <- M %*% K - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% K + lambda*N * diag(N)
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    
    #bb <- M %*% R - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% R
    
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    
    alpha.hat <- solve(AA, bb)
    
    
    # hat e
    term1 <- trn$M2 %*% pre$K
    term2 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term1 + lambda2*N *diag(N), term2)
    e.theta <- solve(trn$L + mu2*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    
    # final estimate
    
    obj <- -mean(e.hat * (R - pre$K %*% alpha.hat)) / mean(e.hat) 
    
    # value <- K %*% alpha.hat
    # 
    # list(alpha = alpha.hat, ehat = e.theta, obj = obj, value = value,
    #      beta = beta.hat)
    list(obj = obj)
    
  }
  
  
  
  ValEst = function(dat, lambda, mu, nZ, nX,
                    theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      L <- trn$L
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      temp <- solve(L+ mu*N*diag(N), L)
      trn$M <- tcrossprod(temp)
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      
    }
    
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    # estimates (depending on tuning parameter)
    #     temp <- solve(L+ mu*N*diag(N), L)
    #     M <- tcrossprod(temp)
    
    term1 <- trn$M %*% pre$K
    
    term2 <- (trn$M %*% Phi) 
    
    term3 <- trn$M %*% R
    
    
    
    
    #AA <- M %*% K - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% K + lambda*N * diag(N)
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    
    #bb <- M %*% R - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% R
    
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    
    alpha.hat <- solve(AA, bb)
    beta.hat <- solve(t(Phi) %*% term2) %*% t(term2)%*% (R - pre$K %*% alpha.hat)
    
    
    
    value <- pre$K %*% alpha.hat
    
    list(alpha = alpha.hat, value = value,
         beta = beta.hat)
    
  }
  
  
  
  ratio_est <- function(dat, lambda, mu, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      stop("error") 
      
    }else{
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      
    }
    
    
    Phi <- trn$Phi
    N <- nrow(dat)
    n <- max(dat$user)
    
    # hat e
    
    term1 <- trn$M2 %*% pre$K
    term2 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term1 + lambda*N *diag(N), term2)
    e.theta <- solve(trn$L + mu*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    
    
    return(list(e.hat = e.hat, e.alpha = e.alpha))
    
  }
  
  
  eta_deriv <- function(theta, dat, lambda, mu, lambda2, mu2, nZ, nX,
                        pre.cal, policy_class){
    
    
    if(is.null(pre.cal)){
      # 
      # trn <- pre_CalMat(dat, nZ, nX)
      # pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # L <- trn$L
      # # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      # temp <- solve(L+ mu*N*diag(N), L)
      # M <- tcrossprod(temp)
      # 
      # temp <- solve(L+ mu2*N*diag(N), L)
      # M2 <- tcrossprod(temp)
      stop("error")
      
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      # M2 <- trn$M2
      
    }
    
    
    
    diff_K <- deriv_of_K(dat, nZ, nX, theta, trn, policy_class = policy_class)
    # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    
    # value
    term1 <- trn$M %*% pre$K
    term2 <- (trn$M %*% Phi) 
    term3 <- trn$M %*% R
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    alpha.hat <- solve(AA, bb)
    
    
    # ratio 
    term21 <- trn$M2 %*% pre$K
    term22 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term21 + lambda2*N *diag(N), term22)
    e.theta <- solve(trn$L + mu2*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    w.hat <- e.hat / mean(e.hat)
    
    
    #compute three terms of gradient]
    p = length(theta)
    
    # (1) ratio 
    
    temp1 <- matrix(0, N, p)
    for(k in 1:p){
      
      temp1[, k] <- trn$M2  %*% (diff_K[, ,k] %*% e.alpha)
      
    }
    deriv_alpha_hat <- -solve(term21 + lambda2*N * diag(N), temp1) 
    
    deriv_ff <- matrix(0, N, p)
    for(k in 1:p){
      
      deriv_ff[, k] <- diff_K[, ,k] %*% e.alpha
      
    }
    deriv_beta <- solve(trn$L+ mu2*N*diag(N), -deriv_ff - pre$K %*%  deriv_alpha_hat) 
    
    # ratio gradient
    # ratio_grad <- ratio_deriv(dat, lambda2, mu2, nZ, nX, theta, trn, diff_K, policy_class = policy_class)
    ratio_grad  <- trn$L %*% deriv_beta
    
    
    norm_ratio_grad <- ratio_grad / mean(e.hat)
    norm_ratio_grad <- norm_ratio_grad - e.hat %*% t(apply(ratio_grad, 2, mean)) / (mean(e.hat) ^ 2)
    
    
    
    # (2) value
    
    deriv_alpha_value = matrix(0, N, p)
    temp_matrix       = matrix(0, N, p)
    for(k in 1:p){
      
      temp_matrix[, k] <- (trn$M - term2 %*% solve(t(Phi) %*%
                                                     term2) %*% t(term2)) %*% (diff_K[, ,k] %*% alpha.hat)
      
      
    }
    
    deriv_alpha <- -solve(AA, temp_matrix)
    
    deriv_v     <- matrix(0, N, p)
    
    for(k in 1:p){
      
      deriv_v[, k] <- diff_K[, , k] %*% alpha.hat
      
    }
    
    value_grad <- deriv_v + pre$K %*% deriv_alpha
    
    # value_grad <- value_deriv(dat, lambda, mu, nZ, nX, theta, trn, diff_K, policy_class = policy_class)
    
    
    
    # final gradient
    final_grad <- t(norm_ratio_grad) %*% (R - pre$K %*% alpha.hat) / N + t(- value_grad) %*% w.hat / N
    
    
    ####make it minimization
    
    final_grad <- -final_grad
    
    
    
    return(final_grad)
    
  }
  
  eta_obj = function(theta, dat, lambda, mu, lambda2, mu2, nZ, nX,
                     pre.cal = NULL, policy_class){
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    result <- RegEst(dat, lambda, mu, lambda2, mu2, nZ, nX,
                     theta, trn, policy_class = policy_class)
    
    output <- result$obj
    
    return(output)
  }  
  
  optim_offoplicy_RL <- function(dat, lambda, mu, lambda2,
                                 mu2, nZ, nX, pre.cal, type_constraint = 1, policy_class){
    
    
    
    if(is.null(pre.cal)){
      
      
      trn <- pre_CalMat(dat, nZ, nX)
      
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    L <- trn$L
    N <- nrow(L)
    # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
    temp <- solve(L+ mu*N*diag(N), L)
    trn$M <- tcrossprod(temp)
    
    temp <- solve(L+ mu2*N*diag(N), L)
    trn$M2 <- tcrossprod(temp)
    
    # remove useless
    rm(temp)
    rm(L)
    
    ###########################################################calculate the length of theta
    
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    
    p <- length(policy_class(X[1, ])) + 1
    
    ##########################################################################
    
    start_time <- Sys.time()
    #Sys.time() - start_time
    
    if(type_constraint == 1){
      
      lower <- -10
      upper <- 10
      
      solution <- lbfgsb3c(rep(0, p), fn = eta_obj, gr= eta_deriv,
                           dat = dat, lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                           nZ = nZ, nX = nX, lower = lower, upper = upper,
                           pre.cal = trn, policy_class = policy_class)
      
      
      
      
    }else{
      
      constraint_bound <- 10
      
      eval_g0 <- function(x,lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                          nZ = nZ, nX = nX, pre.cal = trn, dat = dat) { return( sum(x^2) - constraint_bound )}
      eval_jac_g0 <- function(x, lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                              nZ = nZ, nX = nX, pre.cal = trn, dat = dat) {return( as.matrix(c(2 * x)) )}
      
      
      solution <- nloptr(rep(0, p), eta_obj, eta_deriv,
                         eval_g_ineq = eval_g0,
                         eval_jac_g_ineq = eval_jac_g0,
                         opts = list("algorithm" = "NLOPT_LD_MMA",
                                     "xtol_rel"=1.0e-6),
                         lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                         nZ = nZ, nX = nX, pre.cal = trn, dat = dat)
      
      
    }
    end_time <- Sys.time()
    # 
    print(end_time - start_time)
    
    return(solution)
    
  }
  
  
  #### Setup ####
  
  
  nX <- length(grep("X", colnames(dat)))/2
  nZ <-  length(grep("Z", colnames(dat)))
  
  x.star <- rep(0, nX)
  a.star <- 0
  
  if(is.null(policy_class)){
    
    policy_class = function(x) {
      
      x
      
    }
    
  }
  
  trn <- pre_CalMat(dat, nZ, nX)
  
  N <- nrow(trn$L)
  
  X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
  
  
  p <- length(policy_class(X[1, ])) + 1
  
  
  #### Select tuning parameters ####
  const.all <- c(0.1, 1, 10)
  eta.all <- c(1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
  const.eta.grid <- expand.grid(const = const.all, eta = eta.all)
  # const.eta.grid2 <- const.eta.grid # ratio
  
  
  #number of policy to evaluate
  nP <- 5
  theta_list <- matrix(rnorm(nP * p), nP, p)
  fold.id <- cut(sample(1:nrow(dat), replace = F), breaks=nfold,labels=FALSE)
  
  #     ave_reward <- matrix(0, nrow(const.eta.grid))
  #     ave_reward2 <- matrix(0, nrow(const.eta.grid2))
  
  # error_matrix <- matrix(0, nrow(const.eta.grid), nP)
  # error_matrix2 <- matrix(0, nrow(const.eta.grid2), nP)
  
  error_matrix <- error_matrix2 <-matrix(0, nrow(const.eta.grid), nP)
  
  
  for(k in 1:nP){
    
    
    theta <- theta_list[k, ]  
    
    mat.all <- CalMat(dat, nZ, nX, pre.cal = trn, theta, policy_class = policy_class)
    
    for(i in 1:nfold){
      
      tst.fold <- i;
      trn.dat <- dat[!(fold.id %in% tst.fold), ]
      tst.dat <- dat[fold.id %in% tst.fold, ]
      
      cal.trn <- rapply(trn, 
                        function(x) {if(ncol(x) > 1) {x[!(fold.id %in% tst.fold),!(fold.id %in% tst.fold)]} else{
                          x[!(fold.id %in% tst.fold)]}
                        }, 
                        how = "replace")
      cal.tst <- rapply(trn, 
                        function(x) {if(ncol(x) > 1) {x[(fold.id %in% tst.fold),(fold.id %in% tst.fold)]} else{
                          x[(fold.id %in% tst.fold)]}
                        }, 
                        how = "replace")
      
      cross.K <- mat.all$K[!(fold.id %in% tst.fold), (fold.id %in% tst.fold)]
      
      L <- cal.trn$L
      Ntrain <- nrow(L)
      for(j in 1:nrow(const.eta.grid)){
        
        
        
        eta <- const.eta.grid$eta[j]
        const <- const.eta.grid$const[j]
        
        #         eta2 <- const.eta.grid2$eta[k]
        #         const2 <- const.eta.grid2$const[k]
        
        
        temp <- solve(L+ const*eta*Ntrain*diag(Ntrain), L)
        cal.trn$M <- tcrossprod(temp)
        
        
        
        
        ## value
        
        wm <- ValEst(trn.dat, lambda = eta, mu = const*eta, 
                     nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
        TD.err <- obtainTD.err(cross.K, tst.dat, wm$alpha, wm$beta)
        err <- valid(TD.err, tst.dat, nZ, nX)
        
        #ave_reward[j] = ave_reward[j] + err
        
        error_matrix[j, k] <- error_matrix[j, k] + err
        
        
        
        ## ratio
        cal.trn$M2 <- cal.trn$M
        
        wm <- ratio_est(trn.dat, lambda = eta, mu = const*eta, 
                        nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
        
        TD.err <- obtainTD.err.ratio(cross.K, tst.dat, wm$e.alpha)
        err <- valid(TD.err, tst.dat, nZ, nX)
        #ave_reward2[j] = ave_reward2[j] + err
        
        error_matrix2[j, k] <- error_matrix2[j, k] + err
        
        
      }
      
      
      # for(j in 1:nrow(const.eta.grid2)){
      #   
      #   
      #   
      #   eta <- const.eta.grid2$eta[j]
      #   const <- const.eta.grid2$const[j]
      #   
      #   #         eta2 <- const.eta.grid2$eta[k]
      #   #         const2 <- const.eta.grid2$const[k]
      #   temp <- solve(L+ const*eta*Ntrain*diag(Ntrain), L)
      #   cal.trn$M2 <- tcrossprod(temp)
      #   
      #   
      #   wm <- ratio_est(trn.dat, lambda = eta, mu = const*eta, 
      #                   nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
      #   TD.err <- obtainTD.err.ratio(cross.K, tst.dat, wm$e.alpha)
      #   err <- valid(TD.err, tst.dat, nZ, nX)
      #   #ave_reward2[j] = ave_reward2[j] + err
      #   
      #   error_matrix2[j, k] <- error_matrix2[j, k] + err
      #   
      # }
      # 
      
    }
    
  }
  
  index <- which.min(apply(error_matrix, 1, max))
  index2 <- which.min(apply(error_matrix2, 1, max))
  
  #choose tunning parameter
  
  best_lambda <- const.eta.grid$eta[index]
  
  best_mu <- const.eta.grid$eta[index] * const.eta.grid$const[index]
  
  best_lambda2 <- const.eta.grid$eta[index2]
  
  best_mu2 <- const.eta.grid$eta[index2] * const.eta.grid$const[index2]
  
  
  #### remove unnesssary big datasets to free up memory ####
  rm(cal.trn)
  rm(cal.tst)
  rm(mat.all)
  rm(cross.K)
  rm(L)
  rm(temp)
  
  
  #### Policy learning ####
  solution <- optim_offoplicy_RL(dat, best_lambda, best_mu,best_lambda2, best_mu2,
                                 nZ, nX, pre.cal = trn, type_constraint = type_constraint, policy_class=policy_class)
  
  
  
  
  
  if(type_constraint != 1){
    
    optimal_sol <- solution$solution
    
    optimal_val <- solution$objective
    
    message     <- solution$message
    
    solution <- list()
    
    solution$par <- optimal_sol
    
    solution$value <- optimal_val
    
    solution$message    <- message
    
  }
  
  
  
  solution$lambda_opt <- best_lambda
  
  solution$mu_opt <- best_mu
  
  
  solution$lambda_opt2 <- best_lambda2
  
  solution$mu_opt2 <- best_mu2
  
  # solution$policy_class = policy_class
  
  return(solution)
  
}

PolEvaluation = function(theta, policy_class, ntestrep=100, ntestT= 100, 
                         gen.setup = list(nX = 4, setting = 1, tau = 0.1, fn = function(t) exp(-0.01*(t-1)))){
  
  
  
  eva.pol = function(z, x){
    
    feature <- c(1, policy_class(x))
    prob <- 1 / (1 + exp(-t(feature) %*% theta) ) 
    
    return(prob) 
    
  }
  
  
  
  
  
  test_dat <- DataGen(ntestrep, ntestT, eva.pol, gen.setup)
  
  rwrd.all <- c() # average reward per trajs
  for(i in 1:ntestrep){
    
    rwrd.all <- c(rwrd.all, mean(subset(test_dat, user == i)$reward))
    
  }
  precision <- sd(rwrd.all)/(sqrt(ntestrep))
  
  
  # if(precision > 0.1){
  #   
  #   warning("increase sample size")
  #   
  # }
  optimal_ave_reward <- mean(rwrd.all)
  
  
  
  
  
  return(c(optimal_ave_reward))
}


DataGen = function(n, nT, be.pol = function(z, x) 0.5, 
                   gen.setup = list(nX = 4, setting = 1, tau = 0.2, fn = function(t) exp(-0.05*(t-1)))){
  
  
  stopifnot(gen.setup$setting %in% c(1, 2, 3)) 
  
  if(gen.setup$setting == 1){
    
    # Generative model A (standard MDP setting)
    # ξt ∼ MVN(0, Iq+1),
    # St+1,1 = 0.5 St,1 + 2 ξt,1,
    # St+1,2 = 0.25 St,2 + 0.125 At + 2ξt,2,
    # St+1,3 = 0.9 St,3 + 0.05 St,3At + 0.5 At + ξt,3,
    # St+1,j = 0.25 St,j + ξt,j , j ≥ 3,
    # Rt+1 = 10 − τSt,3 + 0.25 St,1At × (0.04 + 0.02St,1 + 0.02St,2) + 0.16ξt,q+1
    
    nZ <- 1;
    nX <- gen.setup$nX; stopifnot(nX>=3)
    tau <- gen.setup$tau
    
    rwrd.gen = function(z, x, a, x.next) {
      
      10 - tau*x[3] + 0.25 * x[1] * a * (0.04 + 0.02*x[1] + 0.02*x[2]) + 0.16*rnorm(1)
      
    }
    next.X.gen = function(z, x, a) {
      
      
      x1 <- 0.5*x[1] + 2*rnorm(1)
      x2 <- 0.25*x[2] + 0.125*a + 2*rnorm(1)
      x3 <- 0.9*x[3] + 0.05*x[3] * a + 0.5*a + rnorm(1)
      
      x.next <- c(x1, x2, x3)
      
      if(nX > 3){
        
        x.rest <- 0.25 * x[c(4:nX)] + rnorm(nX-3)
        x.next <- c(x.next, x.rest)
      }
      
      return(x.next)
      
      
    }
    
    dat <- NULL;
    for(i in 1:n){
      
      Z <- 1
      next.X <- rnorm(nX);
      
      for(t in 1:nT){
        
        X <- next.X;
        prob <- be.pol(Z, X)
        A <- (runif(1) < prob);
        next.X <- next.X.gen(Z, X, A);
        R <- rwrd.gen(Z, X, A, next.X);
        
        
        dat <- rbind(dat, c(i, t, Z, X, A, prob, R, next.X))
      }
      
    }
    
  }
  
  if(gen.setup$setting == 2){
    
    # Generative model B (non-stationary)
    # [same states model as in A, nonstationary reward]
    # ξt ∼ MVN(0, Iq+1),
    # St+1,1 = 0.5 St,1 + 2 ξt,1,
    # St+1,2 = 0.25 St,2 + 0.125 At + 2ξt,2,
    # St+1,3 = 0.9 St,3 + 0.05 St,3At + 0.5 At + ξt,3,
    # St+1,j = 0.25 St,j + ξt,j , j ≥ 3, 
    # Rt+1 = 10 − τtSt,3 + βtSt,1At(0.04 + 0.02St,1 + 0.02St,2)
    
    nZ <- 1;
    nX <- gen.setup$nX; stopifnot(nX>=3)
    tau <- gen.setup$tau
    time.fn <- gen.setup$fn
    
    rwrd.gen = function(z, x, a, x.next, t) {
      
      10 - tau*x[3] + 0.25 * time.fn(t) * x[1] * a * (0.04 + 0.02*x[1] + 0.02*x[2]) + 0.16*rnorm(1)
      
    }
    next.X.gen = function(z, x, a) {
      
      
      x1 <- 0.5*x[1] + 2*rnorm(1)
      x2 <- 0.25*x[2] + 0.125*a + 2*rnorm(1)
      x3 <- 0.9*x[3] + 0.05*x[3] * a + 0.5*a + rnorm(1)
      
      x.next <- c(x1, x2, x3)
      
      if(nX > 3){
        
        x.rest <- 0.25 * x[c(4:nX)] + rnorm(nX-3)
        x.next <- c(x.next, x.rest)
      }
      
      return(x.next)
      
      
    }
    
    dat <- NULL;
    for(i in 1:n){
      
      Z <- 1
      next.X <- rnorm(nX);
      
      for(t in 1:nT){
        
        X <- next.X;
        prob <- be.pol(Z, X)
        A <- (runif(1) < prob);
        next.X <- next.X.gen(Z, X, A);
        R <- rwrd.gen(Z, X, A, next.X, t);
        
        dat <- rbind(dat, c(i, t, Z, X, A, prob, R, next.X))
      }
      
    }
    
  }
  
  if(gen.setup$setting == 3){
    
    #  Generative model C (non-Markov setting).
    # [same reward as in A but one-lagged state transition]
    # ξt ∼ MVN(0, Iq+1),
    # St+1,1 = 0.5St,1 + 0.25 St−2,1 + 2ξt,1,
    # St+1,2 = 0.25St,2 + 0.125 At + 0.125 St−2,2 + 2ξt,2,
    # St+1,3 = 0.9St,3 + 0.45 St−2,3 + 0.05St,3At + 0.5At + ξt,3, 
    # St+1,j = 0.25St,j + 0.125St−1,j ξt,j , j ≥ 3,
    # Rt+1 = 10 − τSt,3 + 0.25St,1At × (0.04 + 0.02St,1 + 0.02St,2) + 0.16ξt,q+1
    
    nZ <- 1;
    nX <- gen.setup$nX; stopifnot(nX>=3)
    tau <- gen.setup$tau
    
    rwrd.gen = function(z, x, a, x.next) {
      
      10 - tau*x[3] + 0.25 * x[1] * a * (0.04 + 0.02*x[1] + 0.02*x[2]) + 0.16*rnorm(1)
      
    }
    next.X.gen = function(z, x, y, a) {
      
      
      x1 <- 0.5*x[1] + 0.25*y[1] +2*rnorm(1)
      x2 <- 0.25*x[2] + 0.125*a +  0.125*y[2] +2*rnorm(1)
      x3 <- 0.9*x[3] + 0.05*x[3] * a + 0.5*a + 0.05*y[3] + rnorm(1)
      
      x.next <- c(x1, x2, x3)
      
      if(nX > 3){
        
        x.rest <- 0.25 * x[c(4:nX)] + 0.125*y[c(4:nX)] + rnorm(nX-3)
        x.next <- c(x.next, x.rest)
      }
      
      return(x.next)
      
      
      
      
    }
    
    dat <- NULL;
    for(i in 1:n){
      
      Z <- 1
      next.X <- rnorm(nX);
      
      
      for(t in 1:nT){
        
        if(t == 1){
          Y <- rep(0, nX)
        }else{
          Y <- X
        }
        
        X <- next.X;
        
        prob <- be.pol(Z, X)
        A <- (runif(1) < prob);
        next.X <- next.X.gen(Z, X, Y, A);
        R <- rwrd.gen(Z, X, A, next.X);
        
        dat <- rbind(dat, c(i, t, Z, X, A, prob, R, next.X))
      }
      
    }
    
    
  }
  
  colnames(dat) <- c("user", "time", paste("Z", 1:nZ, sep = ""), paste("X", 1:nX, sep = ""), "action", "prob", "reward", paste("next.X", 1:nX, sep = ""))
  
  return(data.frame(dat))
  
  
  
  
}


hs.cv <- function(dat, policy_class=NULL, nfold = 3, type_constraint=1){
  
  
  ##### helper function ####
  k0 = function(z1, x1, z2, x2, sigma = 1) exp(-sum((x1-x2)^2) * sigma)
  
  kernel_train = function(X, Z, A, sigma = 1){
    
    
    
    X <- cbind(X, Z)
    
    temp <- (2 * A - 1) %*% t(2 * A - 1)
    
    #temp[temp == -1] <- 0
    
    #   N <- nrow(X)
    #   
    #   DotProduct <- X %*% t(X)
    #   
    #   DiagDotProduct = as.matrix(diag(DotProduct)) %*% t(as.matrix(rep(1,N)))
    #   
    #   KernelMatrix = DiagDotProduct + t(DiagDotProduct) - 2*DotProduct
    #   
    #   
    #   
    #   KernelMatrix  = exp(-KernelMatrix)
    
    
    
    #median heuristic to select bandwidth
    
    
    
    rbf <- rbfdot(sigma = sigma)
    KK <- kernelMatrix(rbf, X)
    
    
    KK[temp==-1] <- 0
    return(KK)
    
    
  }
  
  kernel_test = function(X, Y, sigma = 1){
    
    N1 <- nrow(X)
    
    N2 <- nrow(Y)
    
    A1 <- X[, ncol(X)]
    A2 <- Y[, ncol(Y)]
    
    X <- X[, 1:(ncol(X)-1)]
    Y <- Y[, 1:(ncol(Y)-1)]
    
    temp <- (2 * A1 - 1) %*% t(2 * A2 - 1)
    
    #temp[temp == -1] <- 0
    #   
    #   DiagDotProductX <- replicate(N2, apply(X * X, 1, sum))
    #   DiagDotProductY1 <- t(replicate(N1, apply(Y * Y, 1, sum)))
    #   DotProductXY1 <- X %*% t(Y)
    #   KK1 <- exp(- (DiagDotProductX + DiagDotProductY1 - 2 * DotProductXY1))  
    #   
    
    rbf <- rbfdot(sigma = sigma)
    KK1 <- kernelMatrix(rbf, X, Y)
    
    
    KK1[temp==-1] <- 0
    
    KK1 <- t(KK1)
    
    
    
    
    return(KK1)                    
  }
  
  shift.Q <-  function(X, Z, A, sigma = 1) {
    
    
    
    
    X1 <- cbind(X, Z, A)
    N <- length(A)
    X2 <- cbind(t(replicate(N, x.star)), Z, a.star)
    
    output <- kernel_train(X, Z, A, sigma) - t(kernel_test(X1, X2, sigma)) - 
      t(kernel_test(X2, X1, sigma)) + t(kernel_test(X2, X2, sigma))
    
    return(output)
    
    
  }
  
  testshift.Q <-  function(X1, Z1, A1, X2, Z2, A2, sigma = 1) {
    
    
    #   k1(z1, x1, a1, z2, x2, a2) - 
    #     k1(z1, x.star, a.star, z2, x2, a2) - 
    #     k1(z1, x1, a1, z2, x.star, a.star) + 
    #     k1(z1, x.star, a.star, z2, x.star, a.star)  
    #   
    
    X3 <- cbind(X1, Z1, A1)
    X4 <- cbind(X2, Z2, A2)
    
    N1 <- length(A1)
    X5 <- cbind(t(replicate(N1, x.star)), Z1, a.star)
    
    N2 <- length(A2)
    X6 <- cbind(t(replicate(N2, x.star)), Z2, a.star)
    
    
    
    output <- t(kernel_test(X3, X4, sigma)) - t(kernel_test(X5, X4, sigma)) -
      t(kernel_test(X3, X6, sigma)) + t(kernel_test(X5, X6, sigma))
    
    return(output)
    
    
  }
  
  
  
  
  
  # Function to calculate shifted kernal
  kernel.Q = function(z1, x1, a1, z2, x2, a2) {
    
    k1(z1, x1, a1, z2, x2, a2) - 
      k1(z1, x.star, a.star, z2, x2, a2) - 
      k1(z1, x1, a1, z2, x.star, a.star) + 
      k1(z1, x.star, a.star, z2, x.star, a.star)  
    
    
  }
  
  # number of baseline covariates Z
  phi = function(z) 1
  
  # kernel between any two cases
  k1 = function(z1, x1, a1, z2, x2, a2, sigma = 1) (a1==a2) * k0(z1, x1, z2, x2, sigma)
  
  
  
  
  # Given the full data, calculate K, L and Phi
  pre_CalMat = function(dat, nZ, nX){
    
    
    
    
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    
    sigma <- 1 / (median(dist(X)))^2
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    
    
    S1 <- t(testshift.Q(X, Z,A, X, Z, matrix(1, N, 1), sigma = sigma))
    S2 <- t(testshift.Q(X, Z,A, X, Z, matrix(0, N, 1),sigma = sigma))
    
    
    
    
    TT <- shift.Q(X, Z, A, sigma = sigma)
    L  <- kernel_train(X, Z, A, sigma = sigma)
    
    
    temp1 <- testshift.Q(X.next, Z, matrix(1, N, 1), X, Z, A, sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d1 <- replicate(N, c(prob1)) * temp1
    
    
    temp2 <- testshift.Q(X.next, Z, matrix(0, N, 1), X, Z, A, sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d2 <- replicate(N, c(1-prob1)) * temp2
    
    
    temp3 <- testshift.Q(X, Z, A, X.next, Z, matrix(1, N, 1), sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d3 <- t(replicate(N, c(prob1))) * (temp3)
    
    
    temp4 <- testshift.Q(X, Z, A, X.next, Z, matrix(0, N, 1), sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d4 <- t(replicate(N, c(1-prob1))) * (temp4)
    
    
    temp5 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(1, N, 1), sigma = sigma)
    
    #d5 <- (prob1 %*% t(prob1)) * temp5 
    
    temp6 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(0, N, 1), sigma = sigma)
    
    #d6 <- (prob1 %*% t(1 - prob1)) * temp6 
    
    temp7 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(1, N, 1), sigma = sigma)
    
    #d7 <- ((1-prob1) %*% t(prob1)) * temp7
    
    temp8 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(0, N, 1), sigma = sigma)
    
    #d8 <- ((1-prob1) %*% t(1-prob1)) * temp8
    
    #K <- TT - (d1 + d2 + d3 + d4) + d5 + d6 + d7 + d8
    
    
    
    list(L = L, Phi = Phi, TT = TT, S1 = S1, S2 = S2,
         temp1 = temp1, temp2 = temp2,temp3 = temp3,temp4 = temp4,
         temp5 = temp5,temp6 = temp6,temp7 = temp7,temp8 = temp8)
    
  }
  
  
  
  
  
  
  
  CalMat = function(dat, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    
    
    
    # temp1 <- trn$temp1
    # temp2 <- trn$temp2
    # temp3 <- trn$temp3
    # temp4 <- trn$temp4
    # temp5 <- trn$temp5
    # temp6 <- trn$temp6
    # temp7 <- trn$temp7
    # temp8 <- trn$temp8
    # L <- trn$L
    # TT <- trn$TT
    # S1 <- trn$S1
    # S2 <- trn$S2
    
    
    
    ########################expand x using policy class###############################
    
    if(p == 2){
      
      policy_X      <- matrix(apply(X, 1, policy_class), ncol = 1)
      policy_X.next <- matrix(apply(X.next, 1, policy_class), ncol = 1)
      
      
    }else{
      
      policy_X      <- t(apply(X, 1, policy_class))
      policy_X.next <- t(apply(X.next, 1, policy_class))
      
      
    }
    
    
    ###############################################################################
    
    
    prob1 <-  (X[, 1] == 1) * (1 / (1 + exp(-cbind(Z, policy_X.next) %*% theta)))
    
    
    d1 <- replicate(N, c(prob1)) * trn$temp1
    
    
    #temp2 <- testshift.Q(X.next, Z, matrix(0, N, 1), X, Z, A)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d2 <- replicate(N, c(1-prob1)) * trn$temp2
    
    
    #temp3 <- testshift.Q(X, Z, A, X.next, Z, matrix(1, N, 1))
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d3 <- t(replicate(N, c(prob1))) * (trn$temp3)
    
    
    #temp4 <- testshift.Q(X, Z, A, X.next, Z, matrix(0, N, 1))
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d4 <- t(replicate(N, c(1-prob1))) * (trn$temp4)
    
    
    #temp5 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(1, N, 1))
    
    d5 <- (prob1 %*% t(prob1)) * trn$temp5 
    
    #temp6 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(0, N, 1))
    
    d6 <- (prob1 %*% t(1 - prob1)) * trn$temp6 
    
    #temp7 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(1, N, 1))
    
    d7 <- ((1-prob1) %*% t(prob1)) * trn$temp7
    
    #temp8 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(0, N, 1))
    
    d8 <- ((1-prob1) %*% t(1-prob1)) * trn$temp8
    
    K <- trn$TT - (d1 + d2 + d3 + d4) + d5 + d6 + d7 + d8
    
    
    
    #   list(L = L, K = K, Phi = Phi, TT = TT, S1 = S1, S2 = S2,
    #        temp1 = temp1, temp2 = temp2,temp3 = temp3,temp4 = temp4,
    #        temp5 = temp5,temp6 = temp6,temp7 = temp7,temp8 = temp8)
    list(K = K)
  }
  
  deriv_of_K = function(dat, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    # temp1 <- trn$temp1
    # temp2 <- trn$temp2
    # temp3 <- trn$temp3
    # temp4 <- trn$temp4
    # temp5 <- trn$temp5
    # temp6 <- trn$temp6
    # temp7 <- trn$temp7
    # temp8 <- trn$temp8
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    p     <- length(theta)
    
    
    
    deriv_KK <- array(0, dim = c(N, N, p))
    
    ########################expand x using policy class###############################
    
    
    if(p == 2){
      
      policy_X      <- matrix(apply(X, 1, policy_class), ncol = 1)
      policy_X.next <- matrix(apply(X.next, 1, policy_class), ncol = 1)
      
      
    }else{
      
      policy_X      <- t(apply(X, 1, policy_class))
      policy_X.next <- t(apply(X.next, 1, policy_class))
      
      
    }
    
    ###############################################################################
    
    
    prob1 <- (X[, 1] == 1) * (1 / (1 + exp(-cbind(Z, policy_X.next) %*% theta)))
    
    d3 <- t(replicate(N, c(prob1))) * (trn$temp3)
    
    d4 <- t(replicate(N, c(1-prob1))) * (trn$temp4)
    
    
    ###################################### we use the policy class
    design_Xnext <- cbind(Z, policy_X.next)
    
    ########################################
    
    
    for(k in 1:p){
      
      
      dd1 <- replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k]))
      
      dd2 <- -replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k]))
      
      dd3 <- t(replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k])))
      
      dd4 <- -t(replicate(N, c(prob1 * (1 - prob1) * design_Xnext[, k])))
      
      dd5 <- (prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(prob1) +
        (prob1) %*% t(prob1* (1 - prob1) * design_Xnext[, k])
      
      dd6 <- (prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(1 - prob1) +
        (prob1) %*% t(-prob1* (1 - prob1) * design_Xnext[, k])
      
      dd7 <- (- prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(prob1) +
        (1 - prob1) %*% t(prob1* (1 - prob1) * design_Xnext[, k])
      
      dd8 <- (- prob1 * (1 - prob1) * design_Xnext[, k]) %*% t(1 - prob1) +
        (1 - prob1) %*% t(- prob1* (1 - prob1) * design_Xnext[, k])
      
      
      
      
      
      deriv_KK[, , k] = -(dd1 * trn$temp1 + dd2 * trn$temp2) +
        -(dd3 * trn$temp3 + dd4 * trn$temp4) +
        dd5 * trn$temp5 + dd6 * trn$temp6 +
        dd7 * trn$temp7 + dd8 * trn$temp8 
      
    }
    
    
    return(deriv_KK)
    
  }
  
  
  obtainTD.err = function(cross.K, tst.dat, alpha, beta){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    if(nZ == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    
    TD.err <- c(tst.dat$reward - Phi %*% beta - t(cross.K) %*% alpha)
    
    
    
    return(TD.err)
    
  }
  
  
  obtainTD.err.ratio = function(cross.K, tst.dat, alpha){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    if(nZ == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    response <- rep(1, length(tst.dat$reward))
    TD.err <- c(response - t(cross.K) %*% alpha)
    
    
    
    return(TD.err)
    
  }
  
  
  valid = function(TD.err, tst.dat, nZ, nX){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(tst.dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    A <- tst.dat$action;
    
    
    # remove the availability for A == 1 (always 1)
    m1 <- gausspr(TD.err[A==1], X[A==1,-1], verbose = F)
    m0 <- gausspr(TD.err[A==0], X[A==0, ], verbose = F)
    
    (mean((predict(m1))^2) + mean((predict(m0))^2))/2 # mean squared estimated Bel.Err
    
  }
  
  
  
  
  
  ###Rregret estimation
  
  RegEst = function(dat, lambda, mu, lambda2, mu2, nZ, nX,
                    theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      L <- trn$L
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      temp <- solve(L+ mu*N*diag(N), L)
      trn$M <- tcrossprod(temp)
      
      temp <- solve(L+ mu2*N*diag(N), L)
      trn$M2 <- tcrossprod(temp)
      
      
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      # M2 <- trn$M2
      
    }
    
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    # estimates (depending on tuning parameter)
    #     temp <- solve(L+ mu*N*diag(N), L)
    #     M <- tcrossprod(temp)
    
    term1 <- trn$M %*% pre$K
    
    term2 <- (trn$M %*% Phi) 
    
    term3 <- trn$M %*% R
    
    
    
    
    #AA <- M %*% K - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% K + lambda*N * diag(N)
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    
    #bb <- M %*% R - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% R
    
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    
    alpha.hat <- solve(AA, bb)
    
    
    # hat e
    term1 <- trn$M2 %*% pre$K
    term2 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term1 + lambda2*N *diag(N), term2)
    e.theta <- solve(trn$L + mu2*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    
    # final estimate
    
    obj <- -mean(e.hat * (R - pre$K %*% alpha.hat)) / mean(e.hat) 
    
    # value <- K %*% alpha.hat
    # 
    # list(alpha = alpha.hat, ehat = e.theta, obj = obj, value = value,
    #      beta = beta.hat)
    list(obj = obj)
    
  }
  
  
  
  ValEst = function(dat, lambda, mu, nZ, nX,
                    theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      L <- trn$L
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      temp <- solve(L+ mu*N*diag(N), L)
      trn$M <- tcrossprod(temp)
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      
    }
    
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    # estimates (depending on tuning parameter)
    #     temp <- solve(L+ mu*N*diag(N), L)
    #     M <- tcrossprod(temp)
    
    term1 <- trn$M %*% pre$K
    
    term2 <- (trn$M %*% Phi) 
    
    term3 <- trn$M %*% R
    
    
    
    
    #AA <- M %*% K - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% K + lambda*N * diag(N)
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    
    #bb <- M %*% R - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% R
    
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    
    alpha.hat <- solve(AA, bb)
    beta.hat <- solve(t(Phi) %*% term2) %*% t(term2)%*% (R - pre$K %*% alpha.hat)
    
    
    
    value <- pre$K %*% alpha.hat
    
    list(alpha = alpha.hat, value = value,
         beta = beta.hat)
    
  }
  
  
  
  ratio_est <- function(dat, lambda, mu, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      stop("error") 
      
    }else{
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      
    }
    
    
    Phi <- trn$Phi
    N <- nrow(dat)
    n <- max(dat$user)
    
    # hat e
    
    term1 <- trn$M2 %*% pre$K
    term2 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term1 + lambda*N *diag(N), term2)
    e.theta <- solve(trn$L + mu*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    
    
    return(list(e.hat = e.hat, e.alpha = e.alpha))
    
  }
  
  
  eta_deriv <- function(theta, dat, lambda, mu, lambda2, mu2, nZ, nX,
                        pre.cal, policy_class){
    
    
    if(is.null(pre.cal)){
      # 
      # trn <- pre_CalMat(dat, nZ, nX)
      # pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # L <- trn$L
      # # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      # temp <- solve(L+ mu*N*diag(N), L)
      # M <- tcrossprod(temp)
      # 
      # temp <- solve(L+ mu2*N*diag(N), L)
      # M2 <- tcrossprod(temp)
      stop("error")
      
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      # M2 <- trn$M2
      
    }
    
    
    
    diff_K <- deriv_of_K(dat, nZ, nX, theta, trn, policy_class = policy_class)
    # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    
    # value
    term1 <- trn$M %*% pre$K
    term2 <- (trn$M %*% Phi) 
    term3 <- trn$M %*% R
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    alpha.hat <- solve(AA, bb)
    
    
    # ratio 
    term21 <- trn$M2 %*% pre$K
    term22 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term21 + lambda2*N *diag(N), term22)
    e.theta <- solve(trn$L + mu2*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    w.hat <- e.hat / mean(e.hat)
    
    
    #compute three terms of gradient]
    p = length(theta)
    
    # (1) ratio 
    
    temp1 <- matrix(0, N, p)
    for(k in 1:p){
      
      temp1[, k] <- trn$M2  %*% (diff_K[, ,k] %*% e.alpha)
      
    }
    deriv_alpha_hat <- -solve(term21 + lambda2*N * diag(N), temp1) 
    
    deriv_ff <- matrix(0, N, p)
    for(k in 1:p){
      
      deriv_ff[, k] <- diff_K[, ,k] %*% e.alpha
      
    }
    deriv_beta <- solve(trn$L+ mu2*N*diag(N), -deriv_ff - pre$K %*%  deriv_alpha_hat) 
    
    # ratio gradient
    # ratio_grad <- ratio_deriv(dat, lambda2, mu2, nZ, nX, theta, trn, diff_K, policy_class = policy_class)
    ratio_grad  <- trn$L %*% deriv_beta
    
    
    norm_ratio_grad <- ratio_grad / mean(e.hat)
    norm_ratio_grad <- norm_ratio_grad - e.hat %*% t(apply(ratio_grad, 2, mean)) / (mean(e.hat) ^ 2)
    
    
    
    # (2) value
    
    deriv_alpha_value = matrix(0, N, p)
    temp_matrix       = matrix(0, N, p)
    for(k in 1:p){
      
      temp_matrix[, k] <- (trn$M - term2 %*% solve(t(Phi) %*%
                                                     term2) %*% t(term2)) %*% (diff_K[, ,k] %*% alpha.hat)
      
      
    }
    
    deriv_alpha <- -solve(AA, temp_matrix)
    
    deriv_v     <- matrix(0, N, p)
    
    for(k in 1:p){
      
      deriv_v[, k] <- diff_K[, , k] %*% alpha.hat
      
    }
    
    value_grad <- deriv_v + pre$K %*% deriv_alpha
    
    # value_grad <- value_deriv(dat, lambda, mu, nZ, nX, theta, trn, diff_K, policy_class = policy_class)
    
    
    
    # final gradient
    final_grad <- t(norm_ratio_grad) %*% (R - pre$K %*% alpha.hat) / N + t(- value_grad) %*% w.hat / N
    
    
    ####make it minimization
    
    final_grad <- -final_grad
    
    
    
    return(final_grad)
    
  }
  
  eta_obj = function(theta, dat, lambda, mu, lambda2, mu2, nZ, nX,
                     pre.cal = NULL, policy_class){
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    result <- RegEst(dat, lambda, mu, lambda2, mu2, nZ, nX,
                     theta, trn, policy_class = policy_class)
    
    output <- result$obj
    
    return(output)
  }  
  
  optim_offoplicy_RL <- function(dat, lambda, mu, lambda2,
                                 mu2, nZ, nX, pre.cal, type_constraint = 1, policy_class){
    
    
    
    if(is.null(pre.cal)){
      
      
      trn <- pre_CalMat(dat, nZ, nX)
      
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    L <- trn$L
    N <- nrow(L)
    # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
    temp <- solve(L+ mu*N*diag(N), L)
    trn$M <- tcrossprod(temp)
    
    temp <- solve(L+ mu2*N*diag(N), L)
    trn$M2 <- tcrossprod(temp)
    
    # remove useless
    rm(temp)
    rm(L)
    
    ###########################################################calculate the length of theta
    
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    
    p <- length(policy_class(X[1, ])) + 1
    
    ##########################################################################
    
    start_time <- Sys.time()
    #Sys.time() - start_time
    
    if(type_constraint == 1){
      
      lower <- -10
      upper <- 10
      
      solution <- lbfgsb3c(rep(0, p), fn = eta_obj, gr= eta_deriv,
                           dat = dat, lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                           nZ = nZ, nX = nX, lower = lower, upper = upper,
                           pre.cal = trn, policy_class = policy_class)
      
      
      
      
    }else{
      
      constraint_bound <- 10
      
      eval_g0 <- function(x,lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                          nZ = nZ, nX = nX, pre.cal = trn, dat = dat) { return( sum(x^2) - constraint_bound )}
      eval_jac_g0 <- function(x, lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                              nZ = nZ, nX = nX, pre.cal = trn, dat = dat) {return( as.matrix(c(2 * x)) )}
      
      
      solution <- nloptr(rep(0, p), eta_obj, eta_deriv,
                         eval_g_ineq = eval_g0,
                         eval_jac_g_ineq = eval_jac_g0,
                         opts = list("algorithm" = "NLOPT_LD_MMA",
                                     "xtol_rel"=1.0e-6),
                         lambda = lambda, mu = mu, lambda2 = lambda2, mu2 = mu2, 
                         nZ = nZ, nX = nX, pre.cal = trn, dat = dat)
      
      
    }
    end_time <- Sys.time()
    # 
    print(end_time - start_time)
    
    return(solution)
    
  }
  
  
  #### Setup ####
  
  
  nX <- length(grep("X", colnames(dat)))/2
  nZ <-  length(grep("Z", colnames(dat)))
  
  x.star <- rep(0, nX)
  a.star <- 0
  
  if(is.null(policy_class)){
    
    policy_class = function(x) {
      
      x
      
    }
    
  }
  
  trn <- pre_CalMat(dat, nZ, nX)
  
  N <- nrow(trn$L)
  
  X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
  
  
  p <- length(policy_class(X[1, ])) + 1
  
  
  #### Select tuning parameters ####
  const.all <- c(0.1, 1, 10)
  eta.all <- c(1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
  const.eta.grid <- expand.grid(const = const.all, eta = eta.all)
  # const.eta.grid2 <- const.eta.grid # ratio
  
  
  #number of policy to evaluate
  nP <- 5
  theta_list <- matrix(rnorm(nP * p), nP, p)
  fold.id <- cut(sample(1:nrow(dat), replace = F), breaks=nfold,labels=FALSE)
  
  #     ave_reward <- matrix(0, nrow(const.eta.grid))
  #     ave_reward2 <- matrix(0, nrow(const.eta.grid2))
  
  # error_matrix <- matrix(0, nrow(const.eta.grid), nP)
  # error_matrix2 <- matrix(0, nrow(const.eta.grid2), nP)
  
  error_matrix <- error_matrix2 <- matrix(0, nrow(const.eta.grid), nP)
  
  
  for(k in 1:nP){
    
    
    theta <- theta_list[k, ]  
    
    mat.all <- CalMat(dat, nZ, nX, pre.cal = trn, theta, policy_class = policy_class)
    
    for(i in 1:nfold){
      
      tst.fold <- i;
      trn.dat <- dat[!(fold.id %in% tst.fold), ]
      tst.dat <- dat[fold.id %in% tst.fold, ]
      
      cal.trn <- rapply(trn, 
                        function(x) {if(ncol(x) > 1) {x[!(fold.id %in% tst.fold),!(fold.id %in% tst.fold)]} else{
                          x[!(fold.id %in% tst.fold)]}
                        }, 
                        how = "replace")
      cal.tst <- rapply(trn, 
                        function(x) {if(ncol(x) > 1) {x[(fold.id %in% tst.fold),(fold.id %in% tst.fold)]} else{
                          x[(fold.id %in% tst.fold)]}
                        }, 
                        how = "replace")
      
      cross.K <- mat.all$K[!(fold.id %in% tst.fold), (fold.id %in% tst.fold)]
      
      L <- cal.trn$L
      Ntrain <- nrow(L)
      for(j in 1:nrow(const.eta.grid)){
        
        
        
        eta <- const.eta.grid$eta[j]
        const <- const.eta.grid$const[j]
        
        #         eta2 <- const.eta.grid2$eta[k]
        #         const2 <- const.eta.grid2$const[k]
        
        
        temp <- solve(L+ const*eta*Ntrain*diag(Ntrain), L)
        cal.trn$M <- tcrossprod(temp)
        
        
        
        
        ## value
        
        wm <- ValEst(trn.dat, lambda = eta, mu = const*eta, 
                     nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
        TD.err <- obtainTD.err(cross.K, tst.dat, wm$alpha, wm$beta)
        err <- valid(TD.err, tst.dat, nZ, nX)
        
        #ave_reward[j] = ave_reward[j] + err
        
        error_matrix[j, k] <- error_matrix[j, k] + err
        
        
        
        ## ratio
        cal.trn$M2 <- cal.trn$M
        
        wm <- ratio_est(trn.dat, lambda = eta, mu = const*eta, 
                        nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
        
        TD.err <- obtainTD.err.ratio(cross.K, tst.dat, wm$e.alpha)
        err <- valid(TD.err, tst.dat, nZ, nX)
        #ave_reward2[j] = ave_reward2[j] + err
        
        error_matrix2[j, k] <- error_matrix2[j, k] + err
        
        
      }
      
      
      # for(j in 1:nrow(const.eta.grid2)){
      #   
      #   
      #   
      #   eta <- const.eta.grid2$eta[j]
      #   const <- const.eta.grid2$const[j]
      #   
      #   #         eta2 <- const.eta.grid2$eta[k]
      #   #         const2 <- const.eta.grid2$const[k]
      #   temp <- solve(L+ const*eta*Ntrain*diag(Ntrain), L)
      #   cal.trn$M2 <- tcrossprod(temp)
      #   
      #   
      #   wm <- ratio_est(trn.dat, lambda = eta, mu = const*eta, 
      #                   nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
      #   TD.err <- obtainTD.err.ratio(cross.K, tst.dat, wm$e.alpha)
      #   err <- valid(TD.err, tst.dat, nZ, nX)
      #   #ave_reward2[j] = ave_reward2[j] + err
      #   
      #   error_matrix2[j, k] <- error_matrix2[j, k] + err
      #   
      # }
      # 
      
    }
    
  }
  
  index <- which.min(apply(error_matrix, 1, max))
  index2 <- which.min(apply(error_matrix2, 1, max))
  
  #choose tunning parameter
  
  best_lambda <- const.eta.grid$eta[index]
  
  best_mu <- const.eta.grid$eta[index] * const.eta.grid$const[index]
  
  best_lambda2 <- const.eta.grid$eta[index2]
  
  best_mu2 <- const.eta.grid$eta[index2] * const.eta.grid$const[index2]
  
  
  #### remove unnesssary big datasets to free up memory ####
  rm(cal.trn)
  rm(cal.tst)
  rm(mat.all)
  rm(cross.K)
  rm(L)
  rm(temp)
  
  
  #### Policy learning ####
  solution <- optim_offoplicy_RL(dat, best_lambda, best_mu,best_lambda2, best_mu2,
                                 nZ, nX, pre.cal = trn, type_constraint = type_constraint, policy_class=policy_class)
  
  
  
  
  
  if(type_constraint != 1){
    
    optimal_sol <- solution$solution
    
    optimal_val <- solution$objective
    
    message     <- solution$message
    
    solution <- list()
    
    solution$par <- optimal_sol
    
    solution$value <- optimal_val
    
    solution$message    <- message
    
  }
  
  
  
  solution$lambda_opt <- best_lambda
  
  solution$mu_opt <- best_mu
  
  
  solution$lambda_opt2 <- best_lambda2
  
  solution$mu_opt2 <- best_mu2
  
  # solution$policy_class = policy_class
  
  return(solution)
  
}



hs.eval <- function(dat, policy_class, theta.eval, nfold = 3){
  
  
  ##### helper function ####
  k0 = function(z1, x1, z2, x2, sigma = 1) exp(-sum((x1-x2)^2) * sigma)
  
  kernel_train = function(X, Z, A, sigma = 1){
    
    
    
    X <- cbind(X, Z)
    
    temp <- (2 * A - 1) %*% t(2 * A - 1)
    
    #temp[temp == -1] <- 0
    
    #   N <- nrow(X)
    #   
    #   DotProduct <- X %*% t(X)
    #   
    #   DiagDotProduct = as.matrix(diag(DotProduct)) %*% t(as.matrix(rep(1,N)))
    #   
    #   KernelMatrix = DiagDotProduct + t(DiagDotProduct) - 2*DotProduct
    #   
    #   
    #   
    #   KernelMatrix  = exp(-KernelMatrix)
    
    
    
    #median heuristic to select bandwidth
    
    
    
    rbf <- rbfdot(sigma = sigma)
    KK <- kernelMatrix(rbf, X)
    
    
    KK[temp==-1] <- 0
    return(KK)
    
    
  }
  
  kernel_test = function(X, Y, sigma = 1){
    
    N1 <- nrow(X)
    
    N2 <- nrow(Y)
    
    A1 <- X[, ncol(X)]
    A2 <- Y[, ncol(Y)]
    
    X <- X[, 1:(ncol(X)-1)]
    Y <- Y[, 1:(ncol(Y)-1)]
    
    temp <- (2 * A1 - 1) %*% t(2 * A2 - 1)
    
    #temp[temp == -1] <- 0
    #   
    #   DiagDotProductX <- replicate(N2, apply(X * X, 1, sum))
    #   DiagDotProductY1 <- t(replicate(N1, apply(Y * Y, 1, sum)))
    #   DotProductXY1 <- X %*% t(Y)
    #   KK1 <- exp(- (DiagDotProductX + DiagDotProductY1 - 2 * DotProductXY1))  
    #   
    
    rbf <- rbfdot(sigma = sigma)
    KK1 <- kernelMatrix(rbf, X, Y)
    
    
    KK1[temp==-1] <- 0
    
    KK1 <- t(KK1)
    
    
    
    
    return(KK1)                    
  }
  
  shift.Q <-  function(X, Z, A, sigma = 1) {
    
    
    
    
    X1 <- cbind(X, Z, A)
    N <- length(A)
    X2 <- cbind(t(replicate(N, x.star)), Z, a.star)
    
    output <- kernel_train(X, Z, A, sigma) - t(kernel_test(X1, X2, sigma)) - 
      t(kernel_test(X2, X1, sigma)) + t(kernel_test(X2, X2, sigma))
    
    return(output)
    
    
  }
  
  testshift.Q <-  function(X1, Z1, A1, X2, Z2, A2, sigma = 1) {
    
    
    #   k1(z1, x1, a1, z2, x2, a2) - 
    #     k1(z1, x.star, a.star, z2, x2, a2) - 
    #     k1(z1, x1, a1, z2, x.star, a.star) + 
    #     k1(z1, x.star, a.star, z2, x.star, a.star)  
    #   
    
    X3 <- cbind(X1, Z1, A1)
    X4 <- cbind(X2, Z2, A2)
    
    N1 <- length(A1)
    X5 <- cbind(t(replicate(N1, x.star)), Z1, a.star)
    
    N2 <- length(A2)
    X6 <- cbind(t(replicate(N2, x.star)), Z2, a.star)
    
    
    
    output <- t(kernel_test(X3, X4, sigma)) - t(kernel_test(X5, X4, sigma)) -
      t(kernel_test(X3, X6, sigma)) + t(kernel_test(X5, X6, sigma))
    
    return(output)
    
    
  }
  
  
  
  
  
  # Function to calculate shifted kernal
  kernel.Q = function(z1, x1, a1, z2, x2, a2) {
    
    k1(z1, x1, a1, z2, x2, a2) - 
      k1(z1, x.star, a.star, z2, x2, a2) - 
      k1(z1, x1, a1, z2, x.star, a.star) + 
      k1(z1, x.star, a.star, z2, x.star, a.star)  
    
    
  }
  
  # number of baseline covariates Z
  phi = function(z) 1
  
  # kernel between any two cases
  k1 = function(z1, x1, a1, z2, x2, a2, sigma = 1) (a1==a2) * k0(z1, x1, z2, x2, sigma)
  
  
  
  
  # Given the full data, calculate K, L and Phi
  pre_CalMat = function(dat, nZ, nX){
    
    
    
    
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    
    sigma <- 1 / (median(dist(X)))^2
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    
    
    S1 <- t(testshift.Q(X, Z,A, X, Z, matrix(1, N, 1), sigma = sigma))
    S2 <- t(testshift.Q(X, Z,A, X, Z, matrix(0, N, 1),sigma = sigma))
    
    
    
    
    TT <- shift.Q(X, Z, A, sigma = sigma)
    L  <- kernel_train(X, Z, A, sigma = sigma)
    
    
    temp1 <- testshift.Q(X.next, Z, matrix(1, N, 1), X, Z, A, sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d1 <- replicate(N, c(prob1)) * temp1
    
    
    temp2 <- testshift.Q(X.next, Z, matrix(0, N, 1), X, Z, A, sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d2 <- replicate(N, c(1-prob1)) * temp2
    
    
    temp3 <- testshift.Q(X, Z, A, X.next, Z, matrix(1, N, 1), sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d3 <- t(replicate(N, c(prob1))) * (temp3)
    
    
    temp4 <- testshift.Q(X, Z, A, X.next, Z, matrix(0, N, 1), sigma = sigma)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    #d4 <- t(replicate(N, c(1-prob1))) * (temp4)
    
    
    temp5 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(1, N, 1), sigma = sigma)
    
    #d5 <- (prob1 %*% t(prob1)) * temp5 
    
    temp6 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(0, N, 1), sigma = sigma)
    
    #d6 <- (prob1 %*% t(1 - prob1)) * temp6 
    
    temp7 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(1, N, 1), sigma = sigma)
    
    #d7 <- ((1-prob1) %*% t(prob1)) * temp7
    
    temp8 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(0, N, 1), sigma = sigma)
    
    #d8 <- ((1-prob1) %*% t(1-prob1)) * temp8
    
    #K <- TT - (d1 + d2 + d3 + d4) + d5 + d6 + d7 + d8
    
    
    
    list(L = L, Phi = Phi, TT = TT, S1 = S1, S2 = S2,
         temp1 = temp1, temp2 = temp2,temp3 = temp3,temp4 = temp4,
         temp5 = temp5,temp6 = temp6,temp7 = temp7,temp8 = temp8)
    
  }
  
  
  
  
  
  
  
  CalMat = function(dat, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    N <- nrow(dat)
    Z <- as.matrix(dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    X.next <- as.matrix(dat[sapply(1:nX, function(i) paste("next.X", i, sep = ""))], N, nX)
    A <- dat$action;
    R <- dat$reward;
    
    # calculation of matrices
    if(length(phi(Z[1, ])) == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    
    
    
    # temp1 <- trn$temp1
    # temp2 <- trn$temp2
    # temp3 <- trn$temp3
    # temp4 <- trn$temp4
    # temp5 <- trn$temp5
    # temp6 <- trn$temp6
    # temp7 <- trn$temp7
    # temp8 <- trn$temp8
    # L <- trn$L
    # TT <- trn$TT
    # S1 <- trn$S1
    # S2 <- trn$S2
    
    
    
    ########################expand x using policy class###############################
    
    if(p == 2){
      
      policy_X      <- matrix(apply(X, 1, policy_class), ncol = 1)
      policy_X.next <- matrix(apply(X.next, 1, policy_class), ncol = 1)
      
      
    }else{
      
      policy_X      <- t(apply(X, 1, policy_class))
      policy_X.next <- t(apply(X.next, 1, policy_class))
      
      
    }
    
    
    ###############################################################################
    
    
    prob1 <-  (X[, 1] == 1) * (1 / (1 + exp(-cbind(Z, policy_X.next) %*% theta)))
    
    
    d1 <- replicate(N, c(prob1)) * trn$temp1
    
    
    #temp2 <- testshift.Q(X.next, Z, matrix(0, N, 1), X, Z, A)
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d2 <- replicate(N, c(1-prob1)) * trn$temp2
    
    
    #temp3 <- testshift.Q(X, Z, A, X.next, Z, matrix(1, N, 1))
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d3 <- t(replicate(N, c(prob1))) * (trn$temp3)
    
    
    #temp4 <- testshift.Q(X, Z, A, X.next, Z, matrix(0, N, 1))
    #prob1 <- exp(cbind(Z, X.next) %*% theta) / (1 + exp(cbind(Z, X.next) %*% theta))
    
    d4 <- t(replicate(N, c(1-prob1))) * (trn$temp4)
    
    
    #temp5 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(1, N, 1))
    
    d5 <- (prob1 %*% t(prob1)) * trn$temp5 
    
    #temp6 <- testshift.Q(X.next, Z, matrix(1, N, 1), X.next, Z, matrix(0, N, 1))
    
    d6 <- (prob1 %*% t(1 - prob1)) * trn$temp6 
    
    #temp7 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(1, N, 1))
    
    d7 <- ((1-prob1) %*% t(prob1)) * trn$temp7
    
    #temp8 <- testshift.Q(X.next, Z, matrix(0, N, 1), X.next, Z, matrix(0, N, 1))
    
    d8 <- ((1-prob1) %*% t(1-prob1)) * trn$temp8
    
    K <- trn$TT - (d1 + d2 + d3 + d4) + d5 + d6 + d7 + d8
    
    
    
    #   list(L = L, K = K, Phi = Phi, TT = TT, S1 = S1, S2 = S2,
    #        temp1 = temp1, temp2 = temp2,temp3 = temp3,temp4 = temp4,
    #        temp5 = temp5,temp6 = temp6,temp7 = temp7,temp8 = temp8)
    list(K = K)
  }
  

  obtainTD.err = function(cross.K, tst.dat, alpha, beta){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    if(nZ == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    
    TD.err <- c(tst.dat$reward - Phi %*% beta - t(cross.K) %*% alpha)
    
    
    
    return(TD.err)
    
  }
  
  
  obtainTD.err.ratio = function(cross.K, tst.dat, alpha){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    if(nZ == 1){
      
      Phi <- matrix(apply(Z, 1, phi), ncol = 1)
      
    }else{
      
      Phi <- t(apply(Z, 1, phi))
    }
    
    response <- rep(1, length(tst.dat$reward))
    TD.err <- c(response - t(cross.K) %*% alpha)
    
    
    
    return(TD.err)
    
  }
  
  
  valid = function(TD.err, tst.dat, nZ, nX){
    
    
    N <- nrow(tst.dat)
    Z <- as.matrix(tst.dat[sapply(1:nZ, function(i) paste("Z", i, sep = ""))], N, nZ)
    X <- as.matrix(tst.dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
    A <- tst.dat$action;
    
    
    # remove the availability for A == 1 (always 1)
    m1 <- gausspr(TD.err[A==1], X[A==1,-1], verbose = F)
    m0 <- gausspr(TD.err[A==0], X[A==0, ], verbose = F)
    
    (mean((predict(m1))^2) + mean((predict(m0))^2))/2 # mean squared estimated Bel.Err
    
  }
  
  
  
  
  
  ###Rregret estimation
  
  RegEst = function(dat, lambda, mu, lambda2, mu2, nZ, nX,
                    theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      L <- trn$L
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      temp <- solve(L+ mu*N*diag(N), L)
      trn$M <- tcrossprod(temp)
      
      temp <- solve(L+ mu2*N*diag(N), L)
      trn$M2 <- tcrossprod(temp)
      
      
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      # M2 <- trn$M2
      
    }
    
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    # estimates (depending on tuning parameter)
    #     temp <- solve(L+ mu*N*diag(N), L)
    #     M <- tcrossprod(temp)
    
    term1 <- trn$M %*% pre$K
    
    term2 <- (trn$M %*% Phi) 
    
    term3 <- trn$M %*% R
    
    
    
    
    #AA <- M %*% K - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% K + lambda*N * diag(N)
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    
    #bb <- M %*% R - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% R
    
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    
    alpha.hat <- solve(AA, bb)
    
    
    # hat e
    term1 <- trn$M2 %*% pre$K
    term2 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term1 + lambda2*N *diag(N), term2)
    e.theta <- solve(trn$L + mu2*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    
    # final estimate
    
    obj <- -mean(e.hat * (R - pre$K %*% alpha.hat)) / mean(e.hat) 
    
    # value <- K %*% alpha.hat
    # 
    # list(alpha = alpha.hat, ehat = e.theta, obj = obj, value = value,
    #      beta = beta.hat)
    list(obj = obj)
    
  }
  
  
  
  ValEst = function(dat, lambda, mu, nZ, nX,
                    theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      L <- trn$L
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      temp <- solve(L+ mu*N*diag(N), L)
      trn$M <- tcrossprod(temp)
    }else{
      
      
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      # M <- trn$M
      
    }
    
    
    
    # L <- trn$L
    # K <- pre$K
    Phi <- trn$Phi
    R <- dat$reward
    
    N <- nrow(dat)
    n <- max(dat$user)
    
    # estimates (depending on tuning parameter)
    #     temp <- solve(L+ mu*N*diag(N), L)
    #     M <- tcrossprod(temp)
    
    term1 <- trn$M %*% pre$K
    
    term2 <- (trn$M %*% Phi) 
    
    term3 <- trn$M %*% R
    
    
    
    
    #AA <- M %*% K - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% K + lambda*N * diag(N)
    
    AA <- term1 - term2 %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term1) + lambda*N * diag(N)
    
    #bb <- M %*% R - M %*% Phi %*% solve(t(Phi) %*% M %*% Phi) %*% t(Phi) %*% M %*% R
    
    bb <- term3 - (term2) %*% solve(t(Phi) %*% term2) %*% (t(Phi) %*% term3)
    
    alpha.hat <- solve(AA, bb)
    beta.hat <- solve(t(Phi) %*% term2) %*% t(term2)%*% (R - pre$K %*% alpha.hat)
    
    
    
    value <- pre$K %*% alpha.hat
    
    list(alpha = alpha.hat, value = value,
         beta = beta.hat)
    
  }
  
  
  
  ratio_est <- function(dat, lambda, mu, nZ, nX, theta, pre.cal = NULL, policy_class){
    
    
    if(is.null(pre.cal)){
      
      stop("error") 
      
    }else{
      
      trn <- pre.cal
      pre <- CalMat(dat, nZ, nX, theta, trn, policy_class = policy_class)
      
    }
    
    
    Phi <- trn$Phi
    N <- nrow(dat)
    n <- max(dat$user)
    
    # hat e
    
    term1 <- trn$M2 %*% pre$K
    term2 <- trn$M2 %*% Phi
    
    e.alpha <- solve(term1 + lambda*N *diag(N), term2)
    e.theta <- solve(trn$L + mu*N*diag(N), Phi - pre$K %*% e.alpha)
    e.hat <- trn$L %*% e.theta
    
    
    return(list(e.hat = e.hat, e.alpha = e.alpha))
    
  }
  
  
  
  eta_obj = function(theta, dat, lambda, mu, lambda2, mu2, nZ, nX,
                     pre.cal = NULL, policy_class){
    
    if(is.null(pre.cal)){
      
      trn <- pre_CalMat(dat, nZ, nX)
      # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
      
    }else{
      
      trn <- pre.cal
      
    }
    
    result <- RegEst(dat, lambda, mu, lambda2, mu2, nZ, nX,
                     theta, trn, policy_class = policy_class)
    
    output <- result$obj
    
    return(output)
  }  
  
 
  
  #### Setup ####
  
  
  nX <- length(grep("X", colnames(dat)))/2
  nZ <-  length(grep("Z", colnames(dat)))
  
  x.star <- rep(0, nX)
  a.star <- 0
  
  if(is.null(policy_class)){
    
    policy_class = function(x) {
      
      x
      
    }
    
  }
  
  trn <- pre_CalMat(dat, nZ, nX)
  
  N <- nrow(trn$L)
  
  X <- as.matrix(dat[sapply(1:nX, function(i) paste("X", i, sep = ""))], N, nX)
  
  
  p <- length(policy_class(X[1, ])) + 1
  
  
  #### Select tuning parameters ####
  const.all <- c(0.1, 1, 10)
  eta.all <- c(1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
  const.eta.grid <- expand.grid(const = const.all, eta = eta.all)
  # const.eta.grid2 <- const.eta.grid # ratio
  
  
  #number of policy to evaluate
  nP <- 1
  theta_list <- matrix(theta.eval, nP, p)
  fold.id <- cut(sample(1:nrow(dat), replace = F), breaks=nfold,labels=FALSE)
  
  #     ave_reward <- matrix(0, nrow(const.eta.grid))
  #     ave_reward2 <- matrix(0, nrow(const.eta.grid2))
  
  # error_matrix <- matrix(0, nrow(const.eta.grid), nP)
  # error_matrix2 <- matrix(0, nrow(const.eta.grid2), nP)
  
  error_matrix <- error_matrix2 <- matrix(0, nrow(const.eta.grid), nP)
  
  
  for(k in 1:nP){
    
    
    theta <- theta_list[k, ]  
    
    mat.all <- CalMat(dat, nZ, nX, pre.cal = trn, theta, policy_class = policy_class)
    
    for(i in 1:nfold){
      
      tst.fold <- i;
      trn.dat <- dat[!(fold.id %in% tst.fold), ]
      tst.dat <- dat[fold.id %in% tst.fold, ]
      
      cal.trn <- rapply(trn, 
                        function(x) {if(ncol(x) > 1) {x[!(fold.id %in% tst.fold),!(fold.id %in% tst.fold)]} else{
                          x[!(fold.id %in% tst.fold)]}
                        }, 
                        how = "replace")
      cal.tst <- rapply(trn, 
                        function(x) {if(ncol(x) > 1) {x[(fold.id %in% tst.fold),(fold.id %in% tst.fold)]} else{
                          x[(fold.id %in% tst.fold)]}
                        }, 
                        how = "replace")
      
      cross.K <- mat.all$K[!(fold.id %in% tst.fold), (fold.id %in% tst.fold)]
      
      L <- cal.trn$L
      Ntrain <- nrow(L)
      for(j in 1:nrow(const.eta.grid)){
        
        
        
        eta <- const.eta.grid$eta[j]
        const <- const.eta.grid$const[j]
        
        #         eta2 <- const.eta.grid2$eta[k]
        #         const2 <- const.eta.grid2$const[k]
        
        
        temp <- solve(L+ const*eta*Ntrain*diag(Ntrain), L)
        cal.trn$M <- tcrossprod(temp)
        
        
        
        
        ## value
        
        wm <- ValEst(trn.dat, lambda = eta, mu = const*eta, 
                     nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
        TD.err <- obtainTD.err(cross.K, tst.dat, wm$alpha, wm$beta)
        err <- valid(TD.err, tst.dat, nZ, nX)
        
        #ave_reward[j] = ave_reward[j] + err
        
        error_matrix[j, k] <- error_matrix[j, k] + err
        
        
        
        ## ratio
        cal.trn$M2 <- cal.trn$M
        
        wm <- ratio_est(trn.dat, lambda = eta, mu = const*eta, 
                        nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
        
        TD.err <- obtainTD.err.ratio(cross.K, tst.dat, wm$e.alpha)
        err <- valid(TD.err, tst.dat, nZ, nX)
        #ave_reward2[j] = ave_reward2[j] + err
        
        error_matrix2[j, k] <- error_matrix2[j, k] + err
        
        
      }
      
      
      # for(j in 1:nrow(const.eta.grid2)){
      #   
      #   
      #   
      #   eta <- const.eta.grid2$eta[j]
      #   const <- const.eta.grid2$const[j]
      #   
      #   #         eta2 <- const.eta.grid2$eta[k]
      #   #         const2 <- const.eta.grid2$const[k]
      #   temp <- solve(L+ const*eta*Ntrain*diag(Ntrain), L)
      #   cal.trn$M2 <- tcrossprod(temp)
      #   
      #   
      #   wm <- ratio_est(trn.dat, lambda = eta, mu = const*eta, 
      #                   nZ = nZ, nX, theta, pre.cal = cal.trn, policy_class = policy_class)
      #   TD.err <- obtainTD.err.ratio(cross.K, tst.dat, wm$e.alpha)
      #   err <- valid(TD.err, tst.dat, nZ, nX)
      #   #ave_reward2[j] = ave_reward2[j] + err
      #   
      #   error_matrix2[j, k] <- error_matrix2[j, k] + err
      #   
      # }
      # 
      
    }
    
  }
  
  index <- which.min(apply(error_matrix, 1, max))
  index2 <- which.min(apply(error_matrix2, 1, max))
  
  #choose tunning parameter
  
  best_lambda <- const.eta.grid$eta[index]
  
  best_mu <- const.eta.grid$eta[index] * const.eta.grid$const[index]
  
  best_lambda2 <- const.eta.grid$eta[index2]
  
  best_mu2 <- const.eta.grid$eta[index2] * const.eta.grid$const[index2]
  
  
  #### remove unnesssary big datasets to free up memory ####
  rm(cal.trn)
  rm(cal.tst)
  rm(mat.all)
  rm(cross.K)
  rm(L)
  rm(temp)
  
  
  #### Policy evaluation ####
  
  
  L <- trn$L
  N <- nrow(L)
  # mat$K.vec.fn(z = 0, x = c(0, 0), a = 1)
  temp <- solve(L+ best_mu*N*diag(N), L)
  trn$M <- tcrossprod(temp)
  
  temp <- solve(L+ best_mu2*N*diag(N), L)
  trn$M2 <- tcrossprod(temp)

  
  
  solution <- eta_obj(theta.eval, dat, best_lambda, best_mu, best_lambda2, best_mu2, nZ, nX,
                     pre.cal = trn, policy_class)
    
  return(solution)
  
}

