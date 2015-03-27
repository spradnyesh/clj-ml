(ns clj-ml.utils
  (:require [incanter.core :as i]
            [clojure.core.matrix.operators :as mo]))

;; elementwise multiple col-vector w/ each column of matrix
;; returns a "vector", not #Vector or #Matrix
(defn vector-mmult [v m]
  (when (= (. v (length)) (. m (getShape 0)))
    (loop [n (. m (getShape 1))
           rslt []]
      (if (zero? n)
        (reverse rslt)
        (recur (dec n)
               (conj rslt (mo/* v (i/$ (dec n) m))))))))
