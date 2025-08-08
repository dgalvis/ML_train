![Conda](https://img.shields.io/badge/conda-25.5.1-blue)
![Python](https://img.shields.io/badge/python-3.13.5-yellow)
![R](https://img.shields.io/badge/R-4.4.3-red)

![Tests](https://github.com/dgalvis/ML_train/actions/workflows/tests.yml/badge.svg)
![pages-build-deployment](https://github.com/dgalvis/ML_train/actions/workflows/pages/pages-build-deployment/badge.svg)

## 📘 Introduction

Welcome! This repository is a personal project designed to **demonstrate machine learning techniques** and explore tools from the **data science** and **software development** ecosystem.

The code and notebooks are structured as hands-on **tutorials**, so feel free to follow along, experiment, and build on them.

To get started, please set up your **virtual environment**. Instructions can be found on the [Requirements](requirements.md) page.


## 🤖 Supervised Learning

**Supervised learning** is a machine learning approach where a model is trained on **labeled data**—that is, input-output pairs. The goal is to learn a general rule that maps inputs to outputs. This type of learning is a form of **predictive modelling**, where models are trained on historical data to make predictions about new, unseen data.


#### 🔍 Examples of Supervised Learning

- **Forecasting energy consumption**  
  A model is trained on a dataset containing daily *energy usage* over the course of a year. It then predicts energy usage for future days based on patterns it has learned.  
  → This is an example of **regression**, because the target output (energy usage) is a **continuous variable**.

- **Email spam detection**  
  A model is trained on a labeled dataset of emails, where each email is marked as *spam* or *not spam*. The model learns to classify new, unseen emails accordingly.  
  → This is an example of **classification**, because the target output is a **categorical variable** (e.g., {"spam", "not spam"} or {0, 1}).

### 📈 Regression

**Regression analysis** refers to a family of methods used to estimate the relationship between a **continuous dependent variable** (also called the output, response, or label) and one or more **independent variables** (also called features, predictors, or regressors).

#### 🔍 Regression Examples

- [Linear Regression](linear_regression.md)
- *(More examples coming soon)*

### 🧮 Classification

**Classification** is the task of predicting **categorical labels** from input data. The model learns to assign new data points to one of a finite set of classes.

#### 🔍 Classification Examples

- [Logistic Regression](logistic_regression.md)
- *(More examples coming soon)*
