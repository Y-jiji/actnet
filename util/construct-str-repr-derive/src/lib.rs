use syn::{*, spanned::Spanned};
use std::str::FromStr;

mod make_body;
use make_body::*;

#[proc_macro_derive(ConstructStrRepr, attributes(mod_path))]
pub fn construct_str_repr_derive(token_stream: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input_type = parse_macro_input!(token_stream as DeriveInput);
    let input_type_name = input_type.ident.to_string();
    let generics_span = input_type.generics.span().unwrap();
    let generics_copy =
        if input_type.generics.gt_token.is_some() { generics_span.source_text().unwrap_or(String::new()) }
        else { String::new() };
    let generics_spec = 
        input_type.generics.const_params().into_iter().map(|x| &x.ident)
        .chain(input_type.generics.type_params().into_iter().map(|x| &x.ident))
        .fold(String::new(), |x, y| x + &y.to_string() + ",");
    let generics_cond = input_type.generics.where_clause.clone()
        .map_or(String::new(), |x| x.span().unwrap().source_text().unwrap_or(String::new()));
    let mut custom_mod_path = None;
    for attr in input_type.attrs {
        if attr.path.is_ident("mod_path") {
            let mod_path: proc_macro2::TokenStream = attr.parse_args().unwrap();
            let mod_path = mod_path.to_string();
            custom_mod_path = Some(mod_path);
        }
    }
    let function_body = make_function_body(
        &input_type_name, &input_type.data, 
        match generics_spec.as_str() { "" => None, _ => Some(format!("::<{generics_spec}>")) }, custom_mod_path
    );
    let expanded = format!("
        impl{generics_copy} ConstructStrRepr for {input_type_name}<{generics_spec}>
        {generics_cond}
        {{
            fn str_repr(self) -> String {{
                {function_body}
            }}
        }}
    ");
    // eprintln!("{expanded}");
    proc_macro::TokenStream::from_str(&expanded).unwrap()
}