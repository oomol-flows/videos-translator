//#region generated meta
type Inputs = {
    file_paths: string[];
};
type Outputs = {
    array: any[];
};
//#endregion

import type { Context } from "@oomol/types/oocana";

export default async function(
    params: Inputs,
    context: Context<Inputs, Outputs>
): Promise<Partial<Outputs> | undefined | void> {

    const videoExtensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'];
    const videoFiles = params.file_paths.filter(filePath => {
        const lowerCaseFilePath = filePath.toLowerCase();
        return videoExtensions.some(ext => lowerCaseFilePath.endsWith(ext));
    });

    return { array: videoFiles };
};