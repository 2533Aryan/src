;-----------------------------------------------------------------------------------------------------------------------
; Name: ARYAN RAVAL
; Project: Generate Art using Neural Networks.
;-----------------------------------------------------------------------------------------------------------------------
; This file contains our Neural Networks that will generate art.
;-----------------------------------------------------------------------------------------------------------------------
; Prerequisite:
; Need "gen-art.math" (math.clj) library for performing math functions.
;-----------------------------------------------------------------------------------------------------------------------
(ns gen-art.network
  (:require [gen-art.math :refer :all]))




;-----------------------------------------------------------------------------------------------------------------------
; layer-record function:
;-----------------------------------------------------------------------------------------------------------------------
(defrecord layer-record [matrix function bias-enabled])

;-----------------------------------------------------------------------------------------------------------------------
; Description:
; This will record shape of our data.
; Basically, each layer have records (means each layer have matrix, function, and bias function).
; Bias function - control the steepness of sigmoid or tangent hyperbolic function.
;-----------------------------------------------------------------------------------------------------------------------




;-----------------------------------------------------------------------------------------------------------------------
; layer function:
;-----------------------------------------------------------------------------------------------------------------------
(defn layer [inputs neurons function bias-enabled]
  (if bias-enabled
    (layer-record. (rand-matrix neurons (+ inputs 1)) function bias-enabled)
    (layer-record. (rand-matrix neurons inputs) function bias-enabled)))

;-----------------------------------------------------------------------------------------------------------------------
; Description:
; This function will create a layer for our neural network.
; If bias-enable is true: it will create new layer (adding 1 to input).
;-----------------------------------------------------------------------------------------------------------------------




;-----------------------------------------------------------------------------------------------------------------------
; feed-forward function:
;-----------------------------------------------------------------------------------------------------------------------
(defn feed-forward [net input]
  (loop [net net input input i 0]
    (if (>= i (count net))
      input
      (let [curr-layer (nth net i)]
        (if (:bias-enabled curr-layer)
          (recur net (map (:function curr-layer)
                          (matrix* (:matrix curr-layer) (concat input [-1]))) (+ i 1))
          (recur net (map (:function curr-layer)
                          (matrix* (:matrix curr-layer) input)) (+ i 1)))))))

;-----------------------------------------------------------------------------------------------------------------------
; Description:
; This function will take our network and inputs.
; If i > our network: pass back our network.
; Otherwise, iterate through each of the layer of our network.
; Base on our current layer, we will check if ":bias-enabled" is true/false.
;-----------------------------------------------------------------------------------------------------------------------




;-----------------------------------------------------------------------------------------------------------------------
; mutate-layer function:
;-----------------------------------------------------------------------------------------------------------------------
(defn mutate-layer [layer-in mutation-chance]
  (if (< (rand) mutation-chance)
    ; n = amount of columns
    ; m = amount of rows
    ; target = random integer (based on amount of columns)
    (let [n (count (:matrix layer-in))
          m (count (nth (:matrix layer-in) 0))
          target (rand-int n)]
      ; create a new layer record
      (layer-record.
       (for [i (range n)]
         (if (= i target)
           ; rand-vec will give a new vector
           (rand-vec m)
           (nth (:matrix layer-in) i)))
       (:function layer-in)
       (:bias-enabled layer-in)))
    layer-in))

;-----------------------------------------------------------------------------------------------------------------------
; Description:
; This function will create a new random matrix (based on the inputs).
; It will also add an element of randomness to make application draw.
;-----------------------------------------------------------------------------------------------------------------------




;-----------------------------------------------------------------------------------------------------------------------
; mutate function:
;-----------------------------------------------------------------------------------------------------------------------
(defn mutate [net mutation-chance]
  (map #(mutate-layer % mutation-chance) net))

;-----------------------------------------------------------------------------------------------------------------------
; Description:
; It is a simple abstraction on top of this mutate-layer function
; This function will take network and mutation chance.
; Then it will map mutate layer function to mutation chance and network.
;-----------------------------------------------------------------------------------------------------------------------




;-----------------------------------------------------------------------------------------------------------------------
; score function:
;-----------------------------------------------------------------------------------------------------------------------
(defn score [net input expected-output]
  (let [output (map #(feed-forward net %) input)
        result (reduce + (map abs-value (flatten (map vec- expected-output output))))]
    [(* result result) net]))
;-----------------------------------------------------------------------------------------------------------------------
; Description:
; This function will take our network, inputs, and expected outputs.
; It will then bind the output to a map calling feed-forward with our network and input in it.
; Then take our expected output and make it a simplified result.
; Pass it back inside a list as a first element and network will be second.
; Basically it will print - how our network is progressing.
;-----------------------------------------------------------------------------------------------------------------------




;-----------------------------------------------------------------------------------------------------------------------
; hill-climb function:
;-----------------------------------------------------------------------------------------------------------------------
(defn hill-climb [net input expected-output iterations]
  (loop [net net
         input input
         expected-output expected-output
         iterations iterations
         i 0]
    (if (= iterations i)
      net
      (let [gen (repeatedly 500 #(mutate net 0.33))
            result (first (sort-by first
                                   (pmap #(score % input expected-output) gen)))]
        (if (= (mod i 10) 0) (println (double (/ i iterations)) (first result)))
        (recur (second result) input expected-output iterations (+ i 1))))))

;-----------------------------------------------------------------------------------------------------------------------
; Description:
; Hill-climb algorithm: It's an optimization techniques
; This function allows us to take an arbitrary number and then incrementally change
; the solution to try and find a better fitting solution.
;-----------------------------------------------------------------------------------------------------------------------



