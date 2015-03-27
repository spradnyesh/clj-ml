(ns clj-ml.ex02
  (:require [clj-ml.utils :as u]
            [incanter.core :as i]
            [incanter.stats :as s]
            [incanter.charts :as c]
            [incanter.io :as io]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mo]))

(defn sigmoid [z]
  (cond (number? z)
        (/ 1 (inc (Math/pow Math/E (- z))))

        (= mikera.vectorz.Vector (type z))
        (map sigmoid (i/to-vect z))

        (= mikera.arrayz.impl.JoinedArray (type z))
        (map #(map sigmoid %) (i/to-vect z))))

(defn- h-theta-x [X theta]
  (sigmoid (m/mmul X theta)))

(defn cost [X y theta]
  (let [h-theta (h-theta-x X theta)]
    (/ (reduce + (mo/+ (mo/* y (map #(Math/log %) h-theta))
                       (mo/* (map #(- 1 %) y) (map (comp #(Math/log %) #(- 1 %)) h-theta))))
       (- m))))

(defn gradient-descent [X y theta alpha num-iters]
  (let [m (. y (getShape 0))]
    (loop [n num-iters
           theta theta]
      (if (zero? n)
        theta
        (let [tmp (u/vector-mmult (mo/- (h-theta-x X theta) y) X)
              sum (map (comp #(/ % m) m/esum #(nth tmp %)) (range (count tmp)))]
          (recur (dec n)
                 (mo/- theta (mo/* alpha sum))))))))

;; uses local gradient-descent, instead of fminunc
;; answers differ greatly :(
(defn non-regularized [alpha n]
  (let [data (i/to-matrix (io/read-dataset "resources/ex2data1.txt"))
        m (i/nrow data)
        y (i/$ (dec (i/ncol data)) data)
        X (->> data
               (i/$ [:not (dec (i/ncol data))]) ; remove "y"
               (i/conj-cols (repeat m 1)) ; including x0
               i/to-matrix)
        theta (i/matrix (repeat (i/ncol X) 0))]
    (gradient-descent X y theta alpha n)))
