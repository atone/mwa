package org.insightcentre.mono.aligners.util;

import java.io.Serializable;

/**
 *
 * @author John McCrae
 */
public class IntPair implements Serializable {
    public final int _1;
    public final int _2;

    public IntPair(int _1, int _2) {
        this._1 = _1;
        this._2 = _2;
    }

    @Override
    public String toString() {
        return "(" + _1 + ", " + _2 + ')';
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 83 * hash + this._1;
        hash = 83 * hash + this._2;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final IntPair other = (IntPair) obj;
        if (this._1 != other._1) {
            return false;
        }
        return this._2 == other._2;
    }
    
}
