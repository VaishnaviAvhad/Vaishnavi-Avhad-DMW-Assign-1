{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ketan Deore.  A-31  DMW_Assignment_1_Q1 : "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Classify digits (0 to 9) using KNN classifier. You can use different values for k neighbors and need to figure out a value of K that gives you a maximum score. You can manually try different values of K or use gridsearchcv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1797, 64)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.datasets import load_digits\n",
    "digits = load_digits()\n",
    "print(digits.data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, ..., 8, 9, 8])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['DESCR', 'data', 'images', 'target', 'target_names']"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dir(digits)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>54</th>\n",
       "      <th>55</th>\n",
       "      <th>56</th>\n",
       "      <th>57</th>\n",
       "      <th>58</th>\n",
       "      <th>59</th>\n",
       "      <th>60</th>\n",
       "      <th>61</th>\n",
       "      <th>62</th>\n",
       "      <th>63</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>...</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows ?? 64 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "    0    1    2     3     4     5    6    7    8    9  ...    54   55   56  \\\n",
       "0  0.0  0.0  5.0  13.0   9.0   1.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0   \n",
       "1  0.0  0.0  0.0  12.0  13.0   5.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0   \n",
       "2  0.0  0.0  0.0   4.0  15.0  12.0  0.0  0.0  0.0  0.0 ...   5.0  0.0  0.0   \n",
       "3  0.0  0.0  7.0  15.0  13.0   1.0  0.0  0.0  0.0  8.0 ...   9.0  0.0  0.0   \n",
       "4  0.0  0.0  0.0   1.0  11.0   0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0   \n",
       "\n",
       "    57   58    59    60    61   62   63  \n",
       "0  0.0  6.0  13.0  10.0   0.0  0.0  0.0  \n",
       "1  0.0  0.0  11.0  16.0  10.0  0.0  0.0  \n",
       "2  0.0  0.0   3.0  11.0  16.0  9.0  0.0  \n",
       "3  0.0  7.0  13.0  13.0   9.0  0.0  0.0  \n",
       "4  0.0  0.0   2.0  16.0   4.0  0.0  0.0  \n",
       "\n",
       "[5 rows x 64 columns]"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.DataFrame(digits.data,digits.target)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>55</th>\n",
       "      <th>56</th>\n",
       "      <th>57</th>\n",
       "      <th>58</th>\n",
       "      <th>59</th>\n",
       "      <th>60</th>\n",
       "      <th>61</th>\n",
       "      <th>62</th>\n",
       "      <th>63</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>20 rows ?? 65 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     0    1     2     3     4     5     6    7    8     9   ...     55   56  \\\n",
       "0  0.0  0.0   5.0  13.0   9.0   1.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "1  0.0  0.0   0.0  12.0  13.0   5.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "2  0.0  0.0   0.0   4.0  15.0  12.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "3  0.0  0.0   7.0  15.0  13.0   1.0   0.0  0.0  0.0   8.0   ...    0.0  0.0   \n",
       "4  0.0  0.0   0.0   1.0  11.0   0.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "5  0.0  0.0  12.0  10.0   0.0   0.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "6  0.0  0.0   0.0  12.0  13.0   0.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "7  0.0  0.0   7.0   8.0  13.0  16.0  15.0  1.0  0.0   0.0   ...    0.0  0.0   \n",
       "8  0.0  0.0   9.0  14.0   8.0   1.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "9  0.0  0.0  11.0  12.0   0.0   0.0   0.0  0.0  0.0   2.0   ...    0.0  0.0   \n",
       "0  0.0  0.0   1.0   9.0  15.0  11.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "1  0.0  0.0   0.0   0.0  14.0  13.0   1.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "2  0.0  0.0   5.0  12.0   1.0   0.0   0.0  0.0  0.0   0.0   ...    2.0  0.0   \n",
       "3  0.0  2.0   9.0  15.0  14.0   9.0   3.0  0.0  0.0   4.0   ...    0.0  0.0   \n",
       "4  0.0  0.0   0.0   8.0  15.0   1.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "5  0.0  5.0  12.0  13.0  16.0  16.0   2.0  0.0  0.0  11.0   ...    0.0  0.0   \n",
       "6  0.0  0.0   0.0   8.0  15.0   1.0   0.0  0.0  0.0   0.0   ...    2.0  0.0   \n",
       "7  0.0  0.0   1.0   8.0  15.0  10.0   0.0  0.0  0.0   3.0   ...    0.0  0.0   \n",
       "8  0.0  0.0  10.0   7.0  13.0   9.0   0.0  0.0  0.0   0.0   ...    0.0  0.0   \n",
       "9  0.0  0.0   6.0  14.0   4.0   0.0   0.0  0.0  0.0   0.0   ...    2.0  0.0   \n",
       "\n",
       "    57    58    59    60    61    62   63  target  \n",
       "0  0.0   6.0  13.0  10.0   0.0   0.0  0.0       0  \n",
       "1  0.0   0.0  11.0  16.0  10.0   0.0  0.0       1  \n",
       "2  0.0   0.0   3.0  11.0  16.0   9.0  0.0       2  \n",
       "3  0.0   7.0  13.0  13.0   9.0   0.0  0.0       3  \n",
       "4  0.0   0.0   2.0  16.0   4.0   0.0  0.0       4  \n",
       "5  0.0   9.0  16.0  16.0  10.0   0.0  0.0       5  \n",
       "6  0.0   1.0   9.0  15.0  11.0   3.0  0.0       6  \n",
       "7  0.0  13.0   5.0   0.0   0.0   0.0  0.0       7  \n",
       "8  0.0  11.0  16.0  15.0  11.0   1.0  0.0       8  \n",
       "9  0.0   9.0  12.0  13.0   3.0   0.0  0.0       9  \n",
       "0  0.0   1.0  10.0  13.0   3.0   0.0  0.0       0  \n",
       "1  0.0   0.0   1.0  13.0  16.0   1.0  0.0       1  \n",
       "2  0.0   3.0  11.0   8.0  13.0  12.0  4.0       2  \n",
       "3  2.0  12.0  12.0  13.0  11.0   0.0  0.0       3  \n",
       "4  0.0   0.0  10.0  15.0   4.0   0.0  0.0       4  \n",
       "5  4.0  15.0  16.0   2.0   0.0   0.0  0.0       5  \n",
       "6  0.0   0.0   7.0  15.0  16.0  11.0  0.0       6  \n",
       "7  0.0   0.0  11.0   9.0   0.0   0.0  0.0       7  \n",
       "8  0.0  11.0  14.0   5.0   0.0   0.0  0.0       8  \n",
       "9  0.0   7.0  16.0  16.0  13.0  11.0  1.0       9  \n",
       "\n",
       "[20 rows x 65 columns]"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['target']=digits.target\n",
    "df.head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "x_train,x_test,y_train,y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3, random_state=10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "->Create KNN classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "knn = KNeighborsClassifier(n_neighbors=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1257"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "540"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "           metric_params=None, n_jobs=None, n_neighbors=3, p=2,\n",
       "           weights='uniform')"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9907407407407407"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn.score(x_test,y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Plot confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[51,  0,  0,  0,  0,  0,  0,  0,  0,  0],\n",
       "       [ 0, 56,  0,  0,  0,  1,  0,  0,  0,  0],\n",
       "       [ 0,  0, 55,  0,  0,  0,  0,  0,  0,  0],\n",
       "       [ 0,  0,  0, 56,  0,  0,  0,  0,  0,  0],\n",
       "       [ 0,  0,  0,  0, 50,  0,  0,  0,  1,  0],\n",
       "       [ 0,  0,  0,  0,  0, 51,  0,  0,  0,  0],\n",
       "       [ 0,  0,  0,  0,  0,  0, 55,  0,  0,  0],\n",
       "       [ 0,  0,  0,  0,  0,  0,  0, 60,  0,  0],\n",
       "       [ 0,  1,  0,  1,  0,  0,  0,  0, 48,  0],\n",
       "       [ 0,  0,  0,  0,  1,  0,  0,  0,  0, 53]], dtype=int64)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "y_prediction = knn.predict(x_test)\n",
    "matrix = confusion_matrix(y_test,y_prediction)\n",
    "matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3. Plot classification report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00        51\n",
      "           1       0.98      0.98      0.98        57\n",
      "           2       1.00      1.00      1.00        55\n",
      "           3       0.98      1.00      0.99        56\n",
      "           4       0.98      0.98      0.98        51\n",
      "           5       0.98      1.00      0.99        51\n",
      "           6       1.00      1.00      1.00        55\n",
      "           7       1.00      1.00      1.00        60\n",
      "           8       0.98      0.96      0.97        50\n",
      "           9       1.00      0.98      0.99        54\n",
      "\n",
      "   micro avg       0.99      0.99      0.99       540\n",
      "   macro avg       0.99      0.99      0.99       540\n",
      "weighted avg       0.99      0.99      0.99       540\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test,y_prediction))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
