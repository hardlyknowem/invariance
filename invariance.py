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
    """An exception to be raised for unexpected errors in validation.

    This is intended to be used when the process of validating itself
    causes an error, not when the validator detects invalid inputs.

    """
    pass


class AbstractValidator(metaclass=abc.ABCMeta):
    """An abstract Validator object.

    A validator is an object that conceptually represents some kind of
    boolean predicate, and provides methods for interacting with that
    predicate in various ways.

    In its most basic use case, a client can use the validator as a
    boolean predicate function, returning the truth-value of the
    predicate for a particular object. This is implemented via the
    `is_valid` method, but also when the object is used as a callable.

    Alternatively, the validator can be used as an assertion with the
    `assertion` method. This method will raise an error (specified by
    the subclass of AbstractValidator) if the predicate is False and
    do nothing otherwise.

    Subclasses of this class should implement the underlying predicate
    using the `validate` method, which returns an exception class if
    the predicate is False and nothing if the predicate is True.

    Since validators conceptually represent boolean predicates, they can
    be combined in much the same ways that real boolean variables can.
    The "&" and "|" ("and" and "or", respectively) operators are defined
    on this class such that, for validators V1 and V2, V1 & V2 will fail
    unless the conditions for both V1 and V2 are met, and V1 | V2 will
    fail only if the conditions for neither V1 nor V2 are met. The "^"
    operator ("exclusive-or") is not supported, as it is unclear what
    exception should be raised if the exclusive-or condition were not
    true.

    The above mix-in methods allow non-validator operands, on either
    side of the operator, to be implicitly converted to BooleanValidator
    objects.

    """
    def is_valid(self, target):
        """Return True if the underlying predicate is true of `target`.

        @param target: The object to validate against the validator.

        """
        return self.validate(target) is None

    def assertion(self, target):
        result = self.validate(target)
        if result is None:
            return
        else:
            raise result

    @abc.abstractmethod
    def validate(self, target):
        """This method implements the validation logic.

        The `target` is the object of which we are asserting an
        invariant. If the invariant is True of the target, this method
        returns None. If the invariant is False, it returns (not raises)
        a fully-constructed instance of an exception.

        @param target: The object to validate against the validator.

        """
        return None

    def __call__(self, target):
        """Dynamic hook to the `is_valid` method.

        @param target: The validation target.

        """
        return self.is_valid(target)

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


#==============================================================================
# Basic Validators
#==============================================================================


class BooleanValidator(AbstractValidator):
    """A Validator object representing its predicate as a callable object.

    This object is constructed from a callable object, which is assumed
    to represent an invariant predicate. The function should return True
    if the predicate obtains and False if not. If the predicate does not
    hold of the validated object, the validate method will return an
    exception with the specified error class and message specified in
    the constructor.

    When the predicate is not True, the `validate` method will construct
    an exception by calling `error_class` with `error_message` as an
    argument. For convenience, the `format` method of `error_message`
    will be called with `target` as a keyword argument. It does not have
    to be used, but it is often helpful.

    """
    DEFAULT_ERROR_MESSAGE = "invariant not true of {target|r}"

    def __init__(self, callable_, error_class=AssertionError,
                 error_message=DEFAULT_ERROR_MESSAGE):
        """Initialize the BooleanValidator.

        @param callable_: A callable object representing a predicate,
            as defined by the class docstring.
        @param error_class: The error to be raised if the validation
            fails.
        @param error_message: A string message for the error class.

        """
        self.callable = callable_
        self.error_class = error_class
        self.error_message = error_message

    def is_valid(self, target):
        """Simplified `is_valid` method to avoid unnecessary overhead.

        @param target: The object to be validated.

        """
        return self.callable(target)

    def validate(self, target):
        """Return the value the underlying predicate would return.

        @param target: The object to validate.

        """
        if not self.callable(target):
            return self.error_class(self.error_message.format(target=target))
        else:
            return None


def Type(type_):
    """Alias for creating a type-checking boolean validator.

    @param type_: A type, or tuple of types, of which the resulting
        validator will require all validation targets to be an instance.

    """
    return BooleanValidator(
        lambda target: isinstance(target, type_),
        TypeError, '{target!r} not of type ' + type_.__name__
    )


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

    def validate(self, target):
        """Return the result of using both validators on `target`.

        If either validator returns an exception, the exception will be
        returned.

        @param target: The object to validate.

        """
        # First, keep in mind we return None unless the invariant is
        # False. So by DeMorgan's law, we actually need an `or` here.
        # The `or` will return the exception if `left` or `right`
        # produces one or `None` if neither do. Also keep in mind that
        # Exception instances are generally always True. If somebody has
        # an Exception subclass with a __bool__ method, then they are
        # doing something weird.
        return self.left(target) or self.right(target)


class DisjunctiveValidator(AbstractValidator):
    """A validator with two children, at least one of which must be true."""
    def __init__(self, left, right):
        """Initialize the DisjunctiveValidator.

        @param left: The first validator to be checked.
        @param right: The second validator to be checked.

        """
        self.left = left
        self.right = right

    def validate(self, target):
        """Return the result of using either validator on `target`.

        If either validator returns `None`, `None` is returned. If
        both validators return exceptions, the exception returned by
        `right` is returned.

        @param target: The object to validate

        """
        # See note about DeMorgan's law in ConjunctiveValidator. The
        # `and` will return `None` if either `left` or `right` return
        # `None`, and an exception if either return an exception. If
        # both return an exception, only the latter will be returned.
        self.left(target) and self.right(target)


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

    def validate(self, target):
        """Validate all the items of `target`.

        @param target: The iterable target to validate.

        """
        for item in target:
            result = self.inner_validator.validate(item)
            if result is not None:
                return result


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

    def validate(self, target):
        """Return True if the elements of `target` are unique.

        @param target: The target to validate.

        """
        keyset = {}
        for item in target:
            key = self.key_function(item)
            inserted = keyset.setdefault(key, item)
            if inserted is not item:
                return ValueError(
                    "{0!r} and {1!r} have same keys".format(inserted, item)
                )
        return None


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

        # Need a special guard singleton, in case `None` is a real element
        first = next(iterator, self._GUARD)
        if first is self._GUARD:
            return

        key = self.key_function(first)

        for item in iterator:
            if self.key_function(item) != key:
                return ValueError(
                    "{0!r} and {1!r} have different keys".format(first, item)
                )
        return None

    _GUARD = object()


#==============================================================================
# Attribute Validation
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
