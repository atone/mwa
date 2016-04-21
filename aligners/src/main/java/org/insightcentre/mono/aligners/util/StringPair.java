package org.insightcentre.mono.aligners.util;

import java.io.Serializable;
import java.util.Objects;

/**
 *
 * @author John McCrae
 */
public class StringPair implements Serializable {
    public final String _1, _2;

    @Override
    public String toString() {
        return "(" + _1 + ", " + _2 + ')';
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 23 * hash + Objects.hashCode(this._1);
        hash = 23 * hash + Objects.hashCode(this._2);
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
        final StringPair other = (StringPair) obj;
        if (!Objects.equals(this._1, other._1)) {
            return false;
        }
        if (!Objects.equals(this._2, other._2)) {
            return false;
        }
        return true;
    }

    public StringPair(String _1, String _2) {
        this._1 = _1;
        this._2 = _2;
    }

    public String toSafeString() {
        StringBuilder sb = new StringBuilder();
        for(char c : _1.toCharArray()) {
            if(Character.isAlphabetic(c)) {
                sb.append(c);
            } else {
                sb.append((int)c);
            }
        }
        sb.append("__");
        for(char c : _2.toCharArray()) {
            if(Character.isAlphabetic(c)) {
                sb.append(c);
            } else {
                sb.append((int)c);
            }
        }
        return sb.toString();
    }
    
}
