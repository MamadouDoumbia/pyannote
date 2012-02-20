#!/usr/bin/env python
# encoding: utf-8

from generic import GenericErrorRate

IER_TOTAL = 'total'
IER_CORRECT = 'correct'
IER_CONFUSION = 'confusion'
IER_FALSE_ALARM = 'false alarm'
IER_MISS = 'miss'
IER_NAME = 'identification error rate'

class IdentificationErrorRate(GenericErrorRate):

    def __init__(self):

        numerator = {IER_CONFUSION: 1., \
                     IER_FALSE_ALARM: 1., \
                     IER_MISS: 1., }
        
        denominator = {IER_TOTAL: 1., }
        other = [IER_CORRECT]
        super(IdentificationErrorRate, self).__init__(IER_NAME, numerator, denominator, other)
    
    
    def __call__(self, reference, hypothesis, detailed=False):
        
        detail = self.initialize()
        
        # common (up-sampled) timeline
        common_timeline = abs(reference.timeline + hypothesis.timeline)
    
        # align reference on common timeline
        R = reference >> common_timeline
    
        # translate and align hypothesis on common timeline
        H = hypothesis >> common_timeline
    
        # loop on all segments
        for segment in common_timeline:
        
            # segment duration
            duration = abs(segment)
        
            # set of IDs in reference segment
            r = R.ids(segment) if segment in R else set([])
            Nr = len(r)
            detail[IER_TOTAL] += duration * Nr
        
            # set of IDs in hypothesis segment
            h = H.ids(segment) if segment in H else set([])
            Nh = len(h)
        
            # number of correct matches
            N_correct = len(r & h)
            detail[IER_CORRECT] += duration * N_correct
        
            # number of incorrect matches
            N_error   = min(Nr, Nh) - N_correct
            detail[IER_CONFUSION] += duration * N_error
        
            # number of misses
            N_miss = max(0, Nr - Nh)
            detail[IER_MISS] += duration * N_miss
        
            # number of false alarms
            N_fa = max(0, Nh - Nr)
            detail[IER_FALSE_ALARM] += duration * N_fa
    
        return self.compute(detail, accumulate=True, detailed=detailed)
        
    def pretty(self, detail):
        
        string = ""
        
        string += "  - duration: %g" % (detail[IER_TOTAL])
        string += "\n"
    
        string += "  - correct: %g" % (detail[IER_CORRECT])
        string += "\n"
    
        string += "  - confusion: %g" % (detail[IER_CONFUSION])
        string += "\n"
        
        string += "  - miss: %g" % (detail[IER_MISS])
        string += "\n"
        
        string += "  - false alarm: %g" % (detail[IER_FALSE_ALARM])
        string += "\n"
    
        string += "  - %s: %g %%" % (self.name, 100*detail[self.name])
        string += "\n"
        
        return string
