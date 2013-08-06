"""Library for asserting invariants of Python objects.

A good programmer is, among other things, a defensive programmer.
Defensive programming techniques prevent software defects, allow defects
to be detected earlier, and even prevent security vulnerabilities. One
simple defensive programming technique is to validate the inputs to
functions and the attributes of classes to fail loudly (i.e., raise an
exception) when these values fail to satisfy their invariants (i.e.,
predicates that must always be true of the value).

While this library does not provide true support for invariants in the
technical sense of the term, it does provide support checking the
invariants of values for an important subset of operations: checking
the parameters of functions, and checking the attributes of classes.

The most important use-case for this library is attribute validation.
(The other parts could be done with assert statements.) This module
provides a descriptor that encapsulates validation for the attributes
of a class, which would otherwise require a lot of boilerplate code
to accomplish with properties. The following is an example:

>>> from invariance import Type, ValidatedAttribute
>>> class Example:
...     id_num = ValidatedAttribute(Type(int) & (lambda val: val >= 0))
...     name = ValidatedAttribute(Type(str) & (lambda s: len(s) < 32))
...     def __init__(self, id_num, name):
...         self.id_num = id_num
...         self.name = name
...
>>> ex1 = Example(100, "my_name")
>>> ex2 = Example(-5, "valid_name")
Traceback (most recent call last):
File "<stdin>", line 1, in ?
ValidationError: invariant not true of -5
>>> ex3 = Example(5, b'valid_name_wrong_type')
Traceback (most recent call last):
File "<stdin>", line 1, in ?
ValidationError: invariant not true of b'valid_name_wrong_type'

Author: Matthew Lefavor
Email:  mclefavor@gmail.com

"""
import abc


class ValidationError(Exception):
    """A generic validation error.

    This is intended to be used as a default error when the user does not
    supply one.

    """
    pass


class AbstractValidator(metaclass=abc.ABCMeta):
    """An abstract Validator object providing mix-ins for validator logic.

    A validator is a callable object accepting a single argument and
    having no return value. Conceptually, a validator must represent
    some kind of boolean predicate. Calling the validator with an object
    as a target should raise a ValidationError if the predicate is not
    true of the object. If the predicate is True, the function should
    have no effect.

    The subclasses should represent their predicates through the `valid`
    method. (An earlier design had the subclasses override the __call__
    method and allowed them to raise any exception; unfortunately, this
    made it difficult to define the compound validators without using
    overly broad exception clauses.) The `valid` method should return
    the truth-value of the predicate this Validator represents for the
    object.

    Since validators conceptually represent boolean predicates, they can
    be combined in much the same ways that real boolean variables can.
    The "&", "|", and "^" ("and", "or", and "xor", respectively)
    operators are defined on this class such that, for validators V1 and
    V2, V1 & V2 will raise an exception unless the conditions for both
    V1 and V2 are met, V1 | V2 will raise an exception only if neither
    condition for neither V1 nor V2 is met, and V1 ^ V2 will raise an
    exception if the conditions for

    (They also allow non-validator operands, on either side, to be
    converted to BooleanValidator objects.)

    """
    @abc.abstractmethod
    def valid(self, item):
        """Return True if the underlying predicate is true of `item`.

        @param item: The object to validate against the validator.

        """
        raise NotImplementedError

    def __call__(self, item):
        """Raise an exception if `item` does not meet the validator condition.

        @param item: The object to validate against the validator.

        """
        if not self.valid(item):
            raise ValidationError("invariant not true of %r" % item)

    def __and__(self, other):
        """Return a Validator combining both operands conjunctively.

        If `other` is not a validator object, then it is assumed to be a
        callable object representing a boolean predicate, taking a
        single parameter and returning True if the predicate obtains of
        that parameter and False if not. The predicate will be converted
        to a BooleanValidator object before being combined with this
        Validator.

        @param other: The other validator, or a callable object, as
            described in the above.

        """
        if not isinstance(other, AbstractValidator):
            other = BooleanValidator(other)
        return ConjunctiveValidator(self, other)

    def __rand__(self, other):
        """Used to implement a mirror of __and__.

        If this is called, it means that `other` has no __and__ method
        defined, and so we can safely assume it is not a Validator
        instance. This method makes the same assumptions of
        non-validators as its non-reflected variant, and will convert
        `other` into a BooleanValidator.

        @param other: A callable object representing a boolean predicate.

        """
        return BooleanValidator(other) & self

    def __or__(self, other):
        """Return a Validator combining both operands disjunctively.

        If `other` is not a validator object, then it is assumed to be a
        callable object representing a boolean predicate, taking a
        single parameter and returning True if the predicate obtains of
        that parameter and False if not. The predicate will be converted
        to a BooleanValidator object before being combined with this
        Validator.

        @param other: The other validator, or a callable object, as
            described in the above.

        """
        if not isinstance(other, AbstractValidator):
            other = BooleanValidator(other)
        return DisjunctiveValidator(self, other)

    def __ror__(self, other):
        """Used to implement mirror of __or__.

        If this is called, it means that `other` has no __and__ method
        defined, and so we can safely assume it is not a Validator
        instance. This method makes the same assumptions of
        non-validators as its non-reflected variant, and will convert
        `other` into a BooleanValidator.

        @param other: A callable object representing a boolean predicate.

        """
        return BooleanValidator(other) | self

    def __xor__(self, other):
        """"Return a Validator combining both operands exclusive-disjunctively.

        If `other` is not a validator object, then it is assumed to be a
        callable object representing a boolean predicate, taking a
        single parameter and returning True if the predicate obtains of
        that parameter and False if not. The predicate will be converted
        to a BooleanValidator object before being combined with this
        Validator.

        @param other: The other validator, or a callable object, as
            described in the above.

        """
        if not isinstance(other, AbstractValidator):
            other = BooleanValidator(other)
        return ExclusiveDisjunctiveValidator(self, other)

    def __rxor__(self, other):
        """Used to implement mirror of __xor__.

        If this is called, it means that `other` has no __and__ method
        defined, and so we can safely assume it is not a Validator
        instance. This method makes the same assumptions of
        non-validators as its non-reflected variant, and will convert
        `other` into a BooleanValidator.

        @param other: A callable object representing a boolean predicate.

        """
        return BooleanValidator(other) ^ self


#==============================================================================
# Basic Validators
#==============================================================================


class BooleanValidator(AbstractValidator):
    """A Validator object representing its predicate as a callable object.

    This object is constructed from a callable object, which is assumed
    to represent an invariant predicate. The function should return True
    if the predicate obtains and False if not. This validator will raise
    an exception if the validation target does not meet the predicate.

    """
    def __init__(self, callable_, error_class=ValidationError):
        """Initialize the BooleanValidator.

        @param callable_: A callable object representing a predicate,
            as defined by the class docstring.
        @param error_class: The error to be raised if the validation
            failse.

        """
        self.callable = callable_
        self.error_class = error_class

    def valid(self, item):
        """Return the value the underlying predicate would return.

        @param item: The item to validate.

        """
        return self.callable(item)


def Type(type_):
    """Alias for creating a type-checking boolean validator.

    @param type_: A type, or tuple of types, of which the resulting
        validator will require all validation targets to be an instance.

    """
    return BooleanValidator(lambda target: isinstance(target, type_))


#==============================================================================
# Compound Validators
#==============================================================================


class ConjunctiveValidator(AbstractValidator):
    """A validator with two children, each of which must be True."""
    def __init__(self, left, right):
        """Initialize the ConjunctiveValidator.

        @param left: The first validator to be checked.
        @param right: The second validator to be checked.

        """
        self.left = left
        self.right = right

    def valid(self, item):
        """Return True if both underlying validators would accept the item.

        @param item: The item to validate.

        """
        return self.left(item) and self.right(item)


class DisjunctiveValidator(AbstractValidator):
    """A validator with two children, at least one of which must be true."""
    def __init__(self, left, right):
        """Initialize the DisjunctiveValidator.

        @param left: The first validator to be checked.
        @param right: The second validator to be checked.

        """
        self.left = left
        self.right = right

    def valid(self, item):
        """Return True if one of the underlying validators would accept `item`.

        @param item: The item to validate.

        """
        return self.left(item) or self.right(item)


class ExclusiveDisjunctiveValidator(AbstractValidator):
    """A validator with two children, exactly one of which must be true."""
    def __init__(self, left, right):
        """Initialize the ExclusiveDisjunctiveValidator.

        @param left: The first validator to be checked.
        @param right: The second validator to be checked.

        """
        self.left = left
        self.right = right

    def valid(self, item):
        """Return True if exactly one of the child validators accepts `item`.

        @param item: The item to validate.

        """
        return self.left(item) ^ self.right(item)


#==============================================================================
# Iterable Validators
#==============================================================================


class ContainerValidator(AbstractValidator):
    """A validator for validating elements in a container."""
    def __init__(self, inner_validator):
        """Initialize the ContainerValidator.

        @param inner_validator: The validator to be applied to each
            element of the validation targets.

        """
        self.inner_validator = inner_validator

    def valid(self, target):
        """Return True if the predicate is True of all elements of `target`.

        @param target: The iterable target to validate.

        """
        for item in target:
            if not self.inner_validator(item):
                return False
        else:
            return True


class UniquenessValidator(AbstractValidator):
    """A validator to assert that all elements of a collection are unique.

    Uniqueness is defined through the use of a function returning a key
    value for items in the collection. An element is unique if and only
    if no other element in the collection shares its key value.

    """
    def __init__(self, key_function):
        """Initialize the UniquenessValidator.

        @param key_function: A function returning a key value for each
            element.

        """
        self.key_function = key_function

    def valid(self, target):
        """Return True if the elements of `target` are unique.

        @param target: The target to validate.

        """
        keyset = set()
        for item in target:
            key = self.key_function(item)
            if key in keyset:
                return False
            keyset.add(key)


class UniformityValidator(AbstractValidator):
    """Asserts that all elements of a collection have a uniform property.

    Uniformity is defined through the use of a function returning a key
    value for items in the collection. A collection is uniform if and
    only if all of its values share the same key value.

    """
    def __init__(self, key_function):
        """Initialize the UniformityValidator.

        @param key_function: A function returning a key value for each
            element.

        """
        self.key_function = key_function

    def valid(self, target):
        """Return True if the elements of `target` are uniform.

        @param target: The target to validate.

        """
        iterator = iter(target)

        # Need a special guard singleton, in case None is a real element
        first = next(iterator, self._GUARD)
        if first is self._GUARD:
            return

        key = self.key_function(first)

        for item in iterator:
            if self.key_function(item) != key:
                raise ValidationError("uniformity not true of %r" % target)
    _GUARD = object()


#==============================================================================
# Iterable Validators
#==============================================================================


class ValidatedAttribute:
    """Descriptor for validating attributes of objects.

    This descriptor encapsulates a common pattern with properties: using
    properties to validate the values of attributes. Using this
    descriptor reduces the amount of boilerplate code that would have
    to be written to write the standard getter and deleter methods.

    An example can be found in the module docstring.

    Under the hood, the ValidatedAttribute descriptor stores objects
    in the object's dictionary. The name will be a non-deterministic
    but unique name starting with a double underscore. If the class
    defines __slots__, then the name can be overridden at class
    definition time.

    """
    def __init__(self, validator, name=None):
        """Initialize the ValidatedAttribute.

        @param validator: The validator to be applied to attribute values.

        """
        self.validator = validator
        self.name = name if name is not None else '__%d' % id(self)

    def __get__(self, instance, owner):
        """Defines attribute access for the descriptor.

        @param instance: The instance whose attribute is being accessed,
            or None if being accessed at the class level.
        @param owner: The class of which this descriptor is an attribute.

        """
        if instance is None:
            return self

        return getattr(instance, self.name)

    def __set__(self, instance, value):
        """Defines attribute modification for the descriptor.

        This is where the actual validation happens.

        @param instance: The instance whose attribute is being modified.
        @param value: The value the attribute should take on.

        """
        self.validator(value)
        setattr(instance, self.name, value)

    def __delete__(self, instance):
        """Defines attribute deletion for the descriptor.

        @param instance: The instance whose attribute is being deleted.

        """
        delattr(instance, self.name)
