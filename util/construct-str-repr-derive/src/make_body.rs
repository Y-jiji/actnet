use syn::*;

pub(crate) 
fn make_function_body(name: &str, data: &Data, _generics: Option<String>, custom_module_path: Option<String>) -> String {
    match data {
        Data::Struct(data_struct) => {
            let (str_part, arg_part, bracket_l, bracket_r) = 
                if data_struct.semi_token.is_some() {
                    let (mut str_part, mut arg_part) = (String::new(), String::new());
                    let idents = data_struct.fields.iter().enumerate().map(|(x, _)| x);
                    for ident in idents {
                        str_part += &format!("{{ }}, ");
                        arg_part += &format!(", self.{ident}.str_repr()");
                    }
                    (str_part, arg_part, "(", ")")
                } else {
                    let (mut str_part, mut arg_part) = (String::new(), String::new());
                    let idents = data_struct.fields.iter().map(|x| x.ident.as_ref().unwrap().to_string());
                    for ident in idents {
                        str_part += &format!("{ident}: {{ }}, ");
                        arg_part += &format!(", self.{ident}.str_repr()");
                    }
                    (str_part, arg_part, "{{", "}}")
                };
            match custom_module_path {
                Some(path) => format!("format!(\"{path}::{name} {bracket_l} {str_part} {bracket_r} \" {arg_part})"),
                None => format!("format!(\"{{ }}::{name} {bracket_l} {str_part} {bracket_r} \", module_path!() {arg_part})")
            }
        }
        Data::Enum(data_enum) => {"match self {".to_string() + &data_enum.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let field_naming = variant.fields
                .iter().next().map(|x| x.ident.is_some())
                .unwrap_or(true);
            let (arm_part, str_part, arg_part, bracket_l_esc, bracket_r_esc, bracket_l, bracket_r) = 
                if !field_naming {
                    let (mut arm_part, mut str_part, mut arg_part) = (String::new(), String::new(), String::new());
                    let idents = variant.fields.iter().enumerate().map(|(x, _)| x);
                    for ident in idents {
                        arm_part += &format!("foo{ident}, ");
                        str_part += &format!("{{ }}, ");
                        arg_part += &format!(", foo{ident}.str_repr()");
                    }
                    (arm_part, str_part, arg_part, "(", ")", "(", ")")
                } else {
                    let (mut arm_part, mut str_part, mut arg_part) = (String::new(), String::new(), String::new());
                    let idents = variant.fields.iter().map(|x| x.ident.as_ref().unwrap().to_string());
                    for ident in idents {
                        arm_part += &format!("{ident}, ");
                        str_part += &format!("{ident}: {{ }}, ");
                        arg_part += &format!(", {ident}.str_repr()");
                    }
                    (arm_part, str_part, arg_part, "{{", "}}", "{", "}")
                };
            match &custom_module_path {
                Some(path) => format!(
                    "Self::{variant_name} {bracket_l} {arm_part} {bracket_r} => format!(\"{path}::{name}::{variant_name} {bracket_l_esc} {str_part} {bracket_r_esc}\" {arg_part})"),
                None => format!(
                    "Self::{variant_name} {bracket_l} {arm_part} {bracket_r} => format!(\"{{}}::{name}::{variant_name} {bracket_l_esc} {str_part} {bracket_r_esc}\", module_path!() {arg_part})"),
            }
        }).fold(String::new(), |x, y| x + &y + ",") + "}"}
        _ => panic!("derive(ConstructStrRepr) for union is not implemented")
    }
}