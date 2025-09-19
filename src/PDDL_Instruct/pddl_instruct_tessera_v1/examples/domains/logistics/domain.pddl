(define (domain logistics)
  (:requirements :strips :typing)
  (:types package airplane truck airport location city)
  (:predicates
    (at ?obj - package ?loc - location)
    (in ?obj - package ?veh)
    (in-city ?loc - location ?c - city)
    (at-airport ?a - airport ?c - city)
    (at-truck ?t - truck ?loc - location)
    (at-plane ?p - airplane ?a - airport))
  (:action load-truck
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (at ?p ?l) (at-truck ?t ?l))
    :effect (and (in ?p ?t) (not (at ?p ?l))))
  (:action unload-truck
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (in ?p ?t) (at-truck ?t ?l))
    :effect (and (at ?p ?l) (not (in ?p ?t))))
  (:action drive
    :parameters (?t - truck ?from - location ?to - location ?c - city)
    :precondition (and (at-truck ?t ?from) (in-city ?from ?c) (in-city ?to ?c))
    :effect (and (at-truck ?t ?to) (not (at-truck ?t ?from))))
  (:action load-plane
    :parameters (?p - package ?a - airport ?plane - airplane)
    :precondition (and (at ?p ?a) (at-plane ?plane ?a))
    :effect (and (in ?p ?plane) (not (at ?p ?a))))
  (:action unload-plane
    :parameters (?p - package ?a - airport ?plane - airplane)
    :precondition (and (in ?p ?plane) (at-plane ?plane ?a))
    :effect (and (at ?p ?a) (not (in ?p ?plane))))
  (:action fly
    :parameters (?plane - airplane ?from - airport ?to - airport)
    :precondition (at-plane ?plane ?from)
    :effect (and (at-plane ?plane ?to) (not (at-plane ?plane ?from)))))
