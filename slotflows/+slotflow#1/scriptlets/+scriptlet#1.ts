//#region generated meta
type Inputs = {
    args: any;
};
type Outputs = {
    destination_folder: string | null;
    source: "en" | "cn" | "ja" | "fr" | "ru" | "de";
    target: "en" | "cn" | "ja" | "fr" | "ru" | "de";
};
//#endregion

export default async function(
    params: Inputs,
    context: Context<Inputs, Outputs>
): Promise<Partial<Outputs> | undefined | void> {
    const {args} = params;
    const destination_folder = args.destination_folder;
    const source = args.source;
    const target = args.target;

    return { destination_folder: destination_folder, source: source, target: target };
};
