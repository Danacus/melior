use std::iter::repeat;

use super::{
    super::{error::Error, utility::sanitize_snake_case_name},
    FieldKind, Operation, OperationField,
};
use convert_case::{Case, Casing};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::GenericArgument;

#[derive(Debug)]
struct TypeStateItem {
    field_name: String,
    generic_param: GenericArgument,
}

impl TypeStateItem {
    pub fn new(field_name: String) -> Self {
        Self {
            generic_param: {
                let ident = format_ident!("__{}", field_name.to_case(Case::Snake));
                syn::parse2(quote!(#ident)).expect("Ident is a valid GenericArgument")
            },
            field_name,
        }
    }
}

#[derive(Debug)]
struct TypeStateList {
    items: Vec<TypeStateItem>,
    unset: GenericArgument,
    set: GenericArgument,
}

impl TypeStateList {
    pub fn new(items: Vec<TypeStateItem>) -> Self {
        Self {
            items,
            unset: syn::parse2(quote!(::melior::dialect::ods_support::Unset)).unwrap(),
            set: syn::parse2(quote!(::melior::dialect::ods_support::Set)).unwrap(),
        }
    }

    pub fn items(&self) -> impl Iterator<Item = &TypeStateItem> {
        self.items.iter()
    }

    pub fn parameters(&self) -> impl Iterator<Item = &GenericArgument> {
        self.items().map(|item| &item.generic_param)
    }

    pub fn parameters_without<'a>(
        &'a self,
        field_name: &'a str,
    ) -> impl Iterator<Item = &GenericArgument> + '_ {
        self.items()
            .filter(move |item| item.field_name != field_name)
            .map(|item| &item.generic_param)
    }

    pub fn arguments_replace<'a>(
        &'a self,
        field_name: &'a str,
        argument: &'a GenericArgument,
    ) -> impl Iterator<Item = &GenericArgument> + '_ {
        self.items().map(move |item| {
            if item.field_name == field_name {
                argument
            } else {
                &item.generic_param
            }
        })
    }

    pub fn arguments_set<'a>(
        &'a self,
        field_name: &'a str,
    ) -> impl Iterator<Item = &GenericArgument> + '_ {
        self.arguments_replace(field_name, &self.set)
    }

    pub fn arguments_unset<'a>(
        &'a self,
        field_name: &'a str,
    ) -> impl Iterator<Item = &GenericArgument> + '_ {
        self.arguments_replace(field_name, &self.unset)
    }

    pub fn arguments_all_set(&self) -> impl Iterator<Item = &GenericArgument> {
        repeat(&self.set).take(self.items.len())
    }

    pub fn arguments_all_unset(&self) -> impl Iterator<Item = &GenericArgument> {
        repeat(&self.unset).take(self.items.len())
    }
}

pub struct OperationBuilder<'o, 'c> {
    operation: &'c Operation<'o>,
    type_state: TypeStateList,
}

impl<'o, 'c> OperationBuilder<'o, 'c> {
    pub fn new(operation: &'c Operation<'o>) -> Result<Self, Error> {
        Ok(Self {
            operation,
            type_state: Self::create_type_state(operation)?,
        })
    }

    pub fn create_builder_fns<'a>(
        &'a self,
        field_names: &'a [Ident],
        phantoms: &'a [TokenStream],
    ) -> impl Iterator<Item = Result<TokenStream, Error>> + 'a {
        let builder_ident = self.builder_identifier();

        self.operation.fields().map(move |field| {
            let name = sanitize_snake_case_name(field.name)?;
            let parameter_type = field.kind.parameter_type()?;
            let argument = quote! { #name: #parameter_type };
            let add = format_ident!("add_{}s", field.kind.as_str());

            // Argument types can be singular and variadic, but add functions in melior
            // are always variadic, so we need to create a slice or vec for singular
            // arguments
            let add_arguments = match &field.kind {
                FieldKind::Element { constraint, .. } => {
                    if constraint.has_variable_length() && !constraint.is_optional() {
                        quote! { #name }
                    } else {
                        quote! { &[#name] }
                    }
                }
                FieldKind::Attribute { .. } => {
                    let name_string = &field.name;

                    quote! {
                        &[(
                            ::melior::ir::Identifier::new(self.context, #name_string),
                            #name.into(),
                        )]
                    }
                }
                FieldKind::Successor { constraint, .. } => {
                    if constraint.is_variadic() {
                        quote! { #name }
                    } else {
                        quote! { &[#name] }
                    }
                }
                FieldKind::Region { constraint, .. } => {
                    if constraint.is_variadic() {
                        quote! { #name }
                    } else {
                        quote! { vec![#name] }
                    }
                }
            };

            Ok(if field.kind.is_optional()? {
                let parameters = self.type_state.parameters().collect::<Vec<_>>();
                quote! {
                    impl<'c, #(#parameters),*> #builder_ident<'c, #(#parameters),*> {
                        pub fn #name(mut self, #argument) -> #builder_ident<'c, #(#parameters),*> {
                            self.builder = self.builder.#add(#add_arguments);
                            self
                        }
                    }
                }
            } else if field.kind.is_result() && self.operation.can_infer_type {
                quote!()
            } else {
                let parameters = self.type_state.parameters_without(field.name);
                let arguments_set = self.type_state.arguments_set(field.name);
                let arguments_unset = self.type_state.arguments_unset(field.name);
                quote! {
                    impl<'c, #(#parameters),*> #builder_ident<'c, #(#arguments_unset),*> {
                        pub fn #name(mut self, #argument) -> #builder_ident<'c, #(#arguments_set),*> {
                            self.builder = self.builder.#add(#add_arguments);
                            let Self { context, mut builder, #(#field_names),* } = self;
                            #builder_ident {
                                context,
                                builder,
                                #(#phantoms),*
                            }
                        }
                    }
                }
            })
        })
    }

    pub fn builder(&self) -> Result<TokenStream, Error> {
        let field_names = self
            .type_state
            .items()
            .map(|field| sanitize_snake_case_name(&field.field_name))
            .collect::<Result<Vec<_>, _>>()?;

        let phantom_fields =
            self.type_state
                .parameters()
                .zip(&field_names)
                .map(|(r#type, name)| {
                    quote! {
                        #name: ::std::marker::PhantomData<#r#type>
                    }
                });

        let phantom_arguments = field_names
            .iter()
            .map(|name| quote! { #name: ::std::marker::PhantomData })
            .collect::<Vec<_>>();

        let builder_fns = self
            .create_builder_fns(&field_names, phantom_arguments.as_slice())
            .collect::<Result<Vec<_>, _>>()?;

        let new = self.create_new_fn(phantom_arguments.as_slice());
        let build = self.create_build_fn();

        let builder_ident = self.builder_identifier();
        let doc = format!("Builder for {}", self.operation.summary);
        let iter_arguments = self.type_state.parameters();

        Ok(quote! {
            #[doc = #doc]
            pub struct #builder_ident <'c, #(#iter_arguments),* > {
                builder: ::melior::ir::operation::OperationBuilder<'c>,
                context: &'c ::melior::Context,
                #(#phantom_fields),*
            }

            #new

            #(#builder_fns)*

            #build
        })
    }

    fn create_build_fn(&self) -> TokenStream {
        let builder_ident = self.builder_identifier();
        let arguments_set = self.type_state.arguments_all_set();
        let class_name = format_ident!("{}", &self.operation.class_name);
        let error = format!("should be a valid {class_name}");
        let maybe_infer = if self.operation.can_infer_type {
            quote! { .enable_result_type_inference() }
        } else {
            quote! {}
        };

        quote! {
            impl<'c> #builder_ident<'c, #(#arguments_set),*> {
                pub fn build(self) -> #class_name<'c> {
                    self.builder #maybe_infer.build().try_into().expect(#error)
                }
            }
        }
    }

    fn create_new_fn(&self, phantoms: &[TokenStream]) -> TokenStream {
        let builder_ident = self.builder_identifier();
        let name = &self.operation.full_name;
        let arguments_unset = self.type_state.arguments_all_unset();

        quote! {
            impl<'c> #builder_ident<'c, #(#arguments_unset),*> {
                pub fn new(location: ::melior::ir::Location<'c>) -> Self {
                    Self {
                        context: unsafe { location.context().to_ref() },
                        builder: ::melior::ir::operation::OperationBuilder::new(#name, location),
                        #(#phantoms),*
                    }
                }
            }
        }
    }

    pub fn create_op_builder_fn(&self) -> TokenStream {
        let builder_ident = self.builder_identifier();
        let arguments_unset = self.type_state.arguments_all_unset();
        quote! {
            pub fn builder(
                location: ::melior::ir::Location<'c>
            ) -> #builder_ident<'c, #(#arguments_unset),*> {
                #builder_ident::new(location)
            }
        }
    }

    pub fn create_default_constructor(&self) -> Result<TokenStream, Error> {
        let class_name = format_ident!("{}", &self.operation.class_name);
        let name = sanitize_snake_case_name(self.operation.short_name)?;
        let arguments = Self::required_fields(self.operation)
            .map(|field| {
                let field = field?;
                let parameter_type = &field.kind.parameter_type()?;
                let parameter_name = &field.sanitized_name;

                Ok(quote! { #parameter_name: #parameter_type })
            })
            .chain([Ok(quote! { location: ::melior::ir::Location<'c> })])
            .collect::<Result<Vec<_>, Error>>()?;
        let builder_calls = Self::required_fields(self.operation)
            .map(|field| {
                let parameter_name = &field?.sanitized_name;

                Ok(quote! { .#parameter_name(#parameter_name) })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let doc = format!("Creates a new {}", self.operation.summary);

        Ok(quote! {
            #[allow(clippy::too_many_arguments)]
            #[doc = #doc]
            pub fn #name<'c>(#(#arguments),*) -> #class_name<'c> {
                #class_name::builder(location)#(#builder_calls)*.build()
            }
        })
    }

    fn required_fields<'a, 'b>(
        operation: &'a Operation<'b>,
    ) -> impl Iterator<Item = Result<&'a OperationField<'b>, Error>> {
        operation
            .fields()
            .filter(|field| !field.kind.is_result() || !operation.can_infer_type)
            .filter_map(|field| match field.kind.is_optional() {
                Ok(optional) => (!optional).then_some(Ok(field)),
                Err(error) => Some(Err(error)),
            })
    }

    fn create_type_state(operation: &'c Operation<'o>) -> Result<TypeStateList, Error> {
        Ok(TypeStateList::new(
            Self::required_fields(operation)
                .map(|field| Ok(TypeStateItem::new(field?.name.to_string())))
                .collect::<Result<_, Error>>()?,
        ))
    }

    fn builder_identifier(&self) -> Ident {
        format_ident!("{}Builder", self.operation.class_name)
    }
}
