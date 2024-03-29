% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{perRow}
\alias{perRow}
\title{Vector times matrix rows}
\usage{
perRow(m, v)
}
\arguments{
\item{m}{numeric rxc matrix}

\item{v}{numeric vector with length c}
}
\value{
rxc matrix with values m[i,j]*v[j]
}
\description{
Multiply a vector to each row of a matrix
Alternative to t(t(m)*v) in pure R
}
