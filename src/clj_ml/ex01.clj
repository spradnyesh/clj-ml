(ns clj-ml.ex01
  (:require [clj-ml.utils :as u]
            [incanter.core :as i]
            [incanter.stats :as s]
            [incanter.charts :as c]
            [incanter.io :as io]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mo]))

(defn- h-theta-x [X theta]
  (m/mmul X theta))

(defn cost [X y theta]
  (let [h-minus-y (mo/- (h-theta-x X theta) y)]
    (/ (m/esum (mo/* h-minus-y h-minus-y)) 2 (.elementCount y))))

(defn gradient-descent [X y theta alpha num-iters]
  (let [m (.elementCount y)]
    (loop [n num-iters
           theta theta]
      (if (zero? n)
        theta
        (let [h-minus-y (mo/- (h-theta-x X theta) y) ; cached for next st
              tmp (u/vector-mmult h-minus-y X) ; cached for next st
              sum (map (comp m/esum #(nth tmp %)) (range (count tmp)))]
          (recur (dec n)
                 (mo/- theta (mo/* alpha (map #(/ % m) sum)))))))))

(defn univariate [alpha n]
  (let [data (i/to-matrix (io/read-dataset "resources/ex1data1.txt"))
        m (i/nrow data)
        X (i/to-matrix (i/conj-cols (repeat m 1) (i/$ 0 data))) ; including x0
        y (i/$ 1 data)
        theta (i/matrix [0 0])]
    (gradient-descent X y theta alpha n)))

(defn feature-normalize [data]
  (loop [n (i/ncol data)
         rslt []]
    (if (zero? n)
      (m/transpose (m/coerce (reverse rslt)))
      (recur (dec n)
             (conj rslt (let [col (i/$ (dec n) data)
                              mean (s/mean col)
                              sd (s/sd col)]
                          (m/div (m/sub col mean) sd)))))))

(defn multivariate [alpha n]
  (let [data (i/to-matrix (io/read-dataset "resources/ex1data2.txt"))
        m (i/nrow data)
        y (i/$ (dec (i/ncol data)) data)
        X (->> data
               (i/$ [:not (dec (i/ncol data))]) ; remove "y"
               feature-normalize
               (i/conj-cols (repeat m 1)) ; including x0
               i/to-matrix)
        theta (i/matrix (repeat (i/ncol X) 0))]
    (gradient-descent X y theta alpha n)))

(defn normal []
  (let [data (i/to-matrix (io/read-dataset "resources/ex1data2.txt"))
        m (i/nrow data)
        y (i/$ (dec (i/ncol data)) data)
        X (->> data
               (i/$ [:not (dec (i/ncol data))]) ; remove "y"
               (i/conj-cols (repeat m 1)) ; including x0
               i/to-matrix)]
    (m/mmul (m/mmul (m/inverse (m/mmul (m/transpose X) X))
                    (m/transpose X))
            y)))
