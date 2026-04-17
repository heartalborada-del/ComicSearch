"""Natural sort order comparator utility.

Implements natural sort order similar to Java NaturalComparator.
Handles mixed numeric and alphabetic strings correctly.
"""


class NaturalComparator:
    """Natural sort order comparator, similar to Java implementation."""

    @staticmethod
    def is_digit(s: str) -> bool:
        """Check if string starts with a digit."""
        return len(s) > 0 and s[0] >= "0" and s[0] <= "9"

    @staticmethod
    def next_slice(s: str, index: int) -> str | None:
        """Get next slice from string at index."""
        length = len(s)
        if index == length:
            return None

        ch = s[index]
        if ch == "." or ch == " ":
            return s[index : index + 1]
        elif "0" <= ch <= "9":
            return s[index : NaturalComparator.next_number_bound(s, index + 1)]
        else:
            return s[index : NaturalComparator.next_other_bound(s, index + 1)]

    @staticmethod
    def next_number_bound(s: str, index: int) -> int:
        """Find the end of a number sequence."""
        length = len(s)
        while index < length:
            ch = s[index]
            if ch < "0" or ch > "9":
                break
            index += 1
        return index

    @staticmethod
    def next_other_bound(s: str, index: int) -> int:
        """Find the end of a non-number, non-separator sequence."""
        length = len(s)
        while index < length:
            ch = s[index]
            if ch == "." or ch == " " or (ch >= "0" and ch <= "9"):
                break
            index += 1
        return index

    @staticmethod
    def remove_leading_zero(s: str) -> str:
        """Remove leading zeros from a number string."""
        if not s:
            return s
        i = 0
        n = len(s) - 1
        while i < n:
            if s[i] != "0":
                return s[i:]
            i += 1
        return s[-1]

    @staticmethod
    def compare_number_string(s1: str, s2: str) -> int:
        """Compare two numeric strings."""
        p1 = NaturalComparator.remove_leading_zero(s1)
        p2 = NaturalComparator.remove_leading_zero(s2)

        l1 = len(p1)
        l2 = len(p2)

        if l1 > l2:
            return 1
        elif l1 < l2:
            return -1
        else:
            for i in range(l1):
                c1 = p1[i]
                c2 = p2[i]
                if c1 > c2:
                    return 1
                elif c1 < c2:
                    return -1

        return -len(s1) + len(s2)

    @staticmethod
    def compare(o1: str | None, o2: str | None) -> int:
        """Compare two strings with natural sort order.

        Args:
            o1: First string to compare
            o2: Second string to compare

        Returns:
            -1 if o1 < o2, 0 if equal, 1 if o1 > o2
        """
        if o1 is None and o2 is None:
            return 0
        if o1 is None:
            return -1
        if o2 is None:
            return 1

        index1 = 0
        index2 = 0
        while True:
            data1 = NaturalComparator.next_slice(o1, index1)
            data2 = NaturalComparator.next_slice(o2, index2)

            if data1 is None and data2 is None:
                return 0
            if data1 is None:
                return -1
            if data2 is None:
                return 1

            index1 += len(data1)
            index2 += len(data2)

            if NaturalComparator.is_digit(data1) and NaturalComparator.is_digit(data2):
                result = NaturalComparator.compare_number_string(data1, data2)
            else:
                # String comparison (case-insensitive)
                result = 0 if data1.lower() == data2.lower() else (1 if data1.lower() > data2.lower() else -1)

            if result != 0:
                return result

